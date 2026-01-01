#!/bin/bash
set -e

echo "🚀 Setting up MCP Smart Environment System..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

# Check if Python version is 3.8 or higher
if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo "❌ Python 3.8+ is required. Found version: $python_version"
    exit 1
fi

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

# Create .env from example if it doesn't exist, or create with defaults
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env file from template (edit if needed)"
    else
        echo "⚠️  .env.example not found. Creating .env with default values..."
        cat > .env << 'EOF'
# MCP Smart Environment System Configuration
# Ollama Configuration
OLLAMA_HOST=http://127.0.0.1:11434

# LLM Model Configuration
MODEL_NAME=qwen2.5:7b

# Database Configuration
DATABASE_PATH=data/smart_environment.db
DATABASE_BACKUP_DIR=data/backups
ENABLE_DATABASE_LOGGING=false

# Feature Flags
USE_COMPACT_PROMPT=false
EOF
        echo "✓ Created .env file with default values (edit if needed)"
    fi
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

