"""
Retrieval module for WheelSense RAG.

This module provides similarity search over the knowledge base using FAISS.
It embeds queries, searches the FAISS index, applies threshold filtering,
and returns structured results.

Features:
- Similarity threshold filtering (default: 0.35) to filter weak matches
- Score gap logic: returns only top result when it significantly outperforms others
- Normalized embeddings for cosine similarity search
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding vector using L2 normalization.
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        Normalized embedding vector (unit vector)
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


class Retriever:
    """
    Retrieval class for performing similarity search over knowledge chunks.
    """
    
    def __init__(self, embeddings_dir: Path = None):
        """
        Initialize the Retriever by loading FAISS index, ID mapping, and embedding model.
        
        Args:
            embeddings_dir: Optional path to embeddings directory. If None, uses default location.
        """
        # Determine paths
        if embeddings_dir is None:
            # Default: assume embeddings/ is sibling to retrieval/
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            embeddings_dir = project_root / "embeddings"
        else:
            embeddings_dir = Path(embeddings_dir)
        
        index_file = embeddings_dir / "faiss_index.bin"
        mapping_file = embeddings_dir / "id_to_chunk.json"
        
        # Load FAISS index
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        print(f"Loading FAISS index from: {index_file}")
        self.index = faiss.read_index(str(index_file))
        
        # Load ID mapping
        if not mapping_file.exists():
            raise FileNotFoundError(f"ID mapping not found: {mapping_file}")
        print(f"Loading ID mapping from: {mapping_file}")
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            self.id_to_chunk = mapping_data['id_to_chunk']
            self.mapping_metadata = mapping_data.get('metadata', {})
        
        # Validate that index and mapping sizes match
        if self.index.ntotal != len(self.id_to_chunk):
            raise ValueError(
                f"Index size ({self.index.ntotal}) does not match "
                f"mapping size ({len(self.id_to_chunk)})"
            )
        
        # Load embedding model
        print("Loading embedding model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✓ Retriever initialized with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.35, score_gap_threshold: float = 0.20) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query: Query string to search for
            top_k: Number of top results to retrieve (default: 3)
            threshold: Minimum similarity score threshold (default: 0.35)
            score_gap_threshold: If gap between top and second result exceeds this,
                               return only the top result (default: 0.20)
            
        Returns:
            Dictionary with either:
            - {"found": true, "chunks": [...]} if results found above threshold
            - {"found": false} if no results above threshold
            
        Notes:
            - If the top result significantly outperforms others (score gap > score_gap_threshold),
              only the top result is returned to reduce noise.
            - Future enhancement: Metadata-based filtering could be added to filter results
              by topic/category tags for improved precision.
        """
        # Validate query
        if not query or not query.strip():
            return {"found": False}
        
        # Embed query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Normalize embedding (matches how chunks were embedded)
        query_embedding = normalize_embedding(query_embedding)
        
        # Reshape to (1, dimension) for FAISS search
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Search FAISS index
        # Returns: (scores, indices) where scores are similarity scores
        scores, indices = self.index.search(query_vector, k=top_k)
        
        # Extract results (FAISS returns shape (1, k))
        scores = scores[0]  # Shape: (k,)
        indices = indices[0]  # Shape: (k,)
        
        # Check threshold: if highest score < threshold, return no results
        if len(scores) == 0 or scores[0] < threshold:
            return {"found": False}
        
        # Build result chunks
        chunks = []
        for i in range(len(scores)):
            # Skip results below threshold
            if scores[i] < threshold:
                continue
            
            # Get FAISS vector ID
            faiss_id = int(indices[i])
            
            # Lookup chunk from ID mapping
            chunk_data = self.id_to_chunk[faiss_id]
            
            # Format result chunk
            chunk_result = {
                "text": chunk_data['text'],
                "score": float(scores[i]),
                "metadata": chunk_data['metadata']
            }
            chunks.append(chunk_result)
        
        # Apply score gap logic: if top result significantly outperforms, return only top
        # This reduces noise when the top result is clearly the best match
        if len(chunks) >= 2:
            score_gap = chunks[0]['score'] - chunks[1]['score']
            if score_gap > score_gap_threshold:
                chunks = [chunks[0]]  # Return only top result
        
        # Return results
        return {
            "found": True,
            "chunks": chunks
        }


def retrieve(query: str, embeddings_dir: Path = None) -> Dict[str, Any]:
    """
    Convenience function for one-off retrieval without instantiating Retriever.
    
    Args:
        query: Query string to search for
        embeddings_dir: Optional path to embeddings directory
        
    Returns:
        Dictionary with retrieval results
    """
    retriever = Retriever(embeddings_dir=embeddings_dir)
    return retriever.retrieve(query)

