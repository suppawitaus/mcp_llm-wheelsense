#!/bin/bash
set -e

echo "🚀 Setting up MCP Smart Environment System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env from example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✓ Created .env file (edit if needed)"
else
    echo "✓ .env file already exists"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/backups
touch data/.gitkeep data/backups/.gitkeep

# Check Ollama
echo "🔍 Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    if ollama list | grep -q "qwen2.5:7b"; then
        echo "✓ Required model (qwen2.5:7b) is installed"
    else
        echo "⚠️  Model qwen2.5:7b not found. Install with: ollama pull qwen2.5:7b"
    fi
else
    echo "⚠️  Ollama not found. Please install from https://ollama.ai"
fi

# Check RAG embeddings
echo "🔍 Checking RAG embeddings..."
if [ -f "rag/embeddings/faiss_index.bin" ] && [ -f "rag/embeddings/id_to_chunk.json" ]; then
    echo "✓ RAG embeddings found"
else
    echo "⚠️  RAG embeddings not found. RAG functionality may not work."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run app.py"
echo "  3. Open browser to http://localhost:8501"

