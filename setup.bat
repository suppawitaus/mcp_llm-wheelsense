@echo off
setlocal enabledelayedexpansion

echo 🚀 Setting up MCP Smart Environment System...

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)
echo ✓ Python found

:: Create virtual environment
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
) else (
    echo ✓ Virtual environment already exists
)

:: Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

:: Create .env from example
if not exist ".env" (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ✓ Created .env file (edit if needed)
) else (
    echo ✓ .env file already exists
)

:: Create data directories
echo 📁 Creating data directories...
if not exist "data" mkdir data
if not exist "data\backups" mkdir data\backups
type nul > data\.gitkeep
type nul > data\backups\.gitkeep

:: Check Ollama (optional - don't fail if not found)
echo 🔍 Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama not found. Please install from https://ollama.ai
) else (
    echo ✓ Ollama is installed
    ollama list | findstr "qwen2.5:7b" >nul 2>&1
    if errorlevel 1 (
        echo ⚠️  Model qwen2.5:7b not found. Install with: ollama pull qwen2.5:7b
    ) else (
        echo ✓ Required model (qwen2.5:7b) is installed
    )
)

:: Check RAG embeddings
echo 🔍 Checking RAG embeddings...
if exist "rag\embeddings\faiss_index.bin" if exist "rag\embeddings\id_to_chunk.json" (
    echo ✓ RAG embeddings found
) else (
    echo ⚠️  RAG embeddings not found. RAG functionality may not work.
)

echo.
echo ✅ Setup complete!
echo.
echo To run the application:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run: streamlit run app.py
echo   3. Open browser to http://localhost:8501

