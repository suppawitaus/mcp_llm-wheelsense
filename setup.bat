@echo off
setlocal enabledelayedexpansion

echo 🚀 Setting up MCP Smart Environment System...

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)

:: Check Python version (must be 3.8+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
:: Validate version: must be 3.8 or higher
if %MAJOR% LSS 3 (
    echo ❌ Python 3.8+ is required. Found version: %PYTHON_VERSION%
    exit /b 1
)
if %MAJOR% EQU 3 (
    :: For Python 3.x, check if minor version is 8 or higher
    :: Use numeric comparison (LSS works correctly for "8", "9", "10", "11", etc.)
    if %MINOR% LSS 8 (
        echo ❌ Python 3.8+ is required. Found version: %PYTHON_VERSION%
        exit /b 1
    )
)
if %MAJOR% GTR 3 (
    :: Python 4+ is fine (future-proof)
    echo ✓ Python found (version %PYTHON_VERSION% - major version %MAJOR%)
)
if %MAJOR% EQU 3 (
    echo ✓ Python found (version %PYTHON_VERSION%)
)

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

:: Create .env from example or create with defaults
if not exist ".env" (
    echo 📝 Creating .env file...
    if exist ".env.example" (
        copy .env.example .env
        echo ✓ Created .env file from template (edit if needed)
    ) else (
        echo ⚠️  .env.example not found. Creating .env with default values...
        (
            echo # MCP Smart Environment System Configuration
            echo # Ollama Configuration
            echo OLLAMA_HOST=http://127.0.0.1:11434
            echo.
            echo # LLM Model Configuration
            echo MODEL_NAME=qwen2.5:7b
            echo.
            echo # Database Configuration
            echo DATABASE_PATH=data/smart_environment.db
            echo DATABASE_BACKUP_DIR=data/backups
            echo ENABLE_DATABASE_LOGGING=false
            echo.
            echo # Feature Flags
            echo USE_COMPACT_PROMPT=false
        ) > .env
        echo ✓ Created .env file with default values (edit if needed)
    )
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

