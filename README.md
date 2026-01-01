# MCP Smart Environment System

A Streamlit-based smart environment assistant for elderly/disabled users. The system uses LLM (via Ollama) to control devices, manage schedules, and provide health-related guidance through RAG (Retrieval-Augmented Generation).

## Features

- **Device Control**: Control lights, AC, TV, Fan, and Alarm across multiple rooms
- **Schedule Management**: Add, modify, and delete daily schedule items
- **Health Knowledge Base**: RAG system provides tailored health recommendations based on user conditions
- **Notification System**: Proactive house checks and activity reminders
- **Natural Language Interface**: Chat-based interaction powered by LLM

## Project Structure

```
mcp_llm/
├── app.py                    # Streamlit UI entry point
├── config.py                 # Configuration constants
├── requirements.txt          # Python dependencies
├── core/                     # Core business logic
│   ├── state.py             # State management
│   └── activity_derivation.py
├── mcp/                      # MCP protocol components
│   ├── server.py            # MCP server implementation
│   └── router.py            # Tool call router
├── llm/                      # LLM interaction
│   ├── client.py            # LLM client
│   └── prompts.py           # System prompts
├── services/                  # Application services
│   └── notification.py      # Notification service
├── utils/                     # Utilities
│   └── safety_logger.py     # Safety logging
└── rag/                      # RAG system
    ├── data/chunks/         # Health knowledge chunks
    ├── embeddings/          # FAISS index and mappings
    └── retrieval/
        └── retriever.py    # RAG retriever
```

## Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
   - Download from: https://ollama.ai
   - Required model: `qwen2.5:7b` (or as configured)

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd mcp_llm

# Run setup script (recommended)
# Linux/Mac:
./setup.sh

# Windows:
setup.bat

# Or manual setup:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment (copy and edit)
cp .env.example .env
# Edit .env with your settings (optional - defaults work for localhost)

# Verify Ollama is running
ollama list  # Should show qwen2.5:7b

# Run the application
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Configuration

Configuration is done via environment variables. Copy `.env.example` to `.env` and customize:

- `OLLAMA_HOST`: Ollama server URL (default: http://127.0.0.1:11434)
- `MODEL_NAME`: LLM model name (default: qwen2.5:7b)
- `DATABASE_PATH`: Path to SQLite database (default: data/smart_environment.db)
- `ENABLE_DATABASE_LOGGING`: Enable SQL query logging (default: false)
- `USE_COMPACT_PROMPT`: Use compact prompt format (default: false)

See `.env.example` for all available options.

For advanced configuration (rooms, devices, default location), edit `config.py` directly.

## RAG System Setup

The RAG system requires pre-generated embeddings. The embeddings should be in `rag/embeddings/`:
- `faiss_index.bin`: FAISS vector index
- `id_to_chunk.json`: ID to chunk mapping

These files should be included in the repository. If they are missing, RAG functionality will not work.

## Usage

1. **Start the app** (see Running the Application)

2. **Interact via chat**:
   - Ask questions: "What devices are on?"
   - Control devices: "Turn on the light"
   - Manage schedule: "I have a meeting at 14:00"
   - Get health advice: "What should I eat for breakfast?" (if user has health conditions)

3. **Room Map**: Click on rooms in the left panel to change user location

4. **Schedule**: View and manage daily schedule items

## Key Components

### LLM Client (`llm/client.py`)
- Handles communication with Ollama
- Parses LLM responses into tool calls
- Manages conversation context and summarization

### MCP Server (`mcp/server.py`)
- Implements MCP protocol tools
- Executes device control and schedule modifications
- Integrates with RAG system for health queries

### State Manager (`core/state.py`)
- Manages device states, user location, and schedules
- Handles schedule cloning and one-time events

### RAG System (`rag/`)
- Retrieves relevant health knowledge based on user queries
- Uses FAISS for similarity search
- Provides context-aware health recommendations

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Verify all dependencies: `pip list`
- Check you're in project root directory
- Reinstall dependencies: `pip install -r requirements.txt`

### Ollama Connection Issues
- Verify Ollama is running: `ollama list`
- Check `OLLAMA_HOST` in `.env` (or `config.py`) matches your Ollama instance
- Ensure the model is installed: `ollama pull qwen2.5:7b`
- Test connection: `curl http://127.0.0.1:11434/api/tags`

### RAG System Not Working
- Verify `rag/embeddings/faiss_index.bin` exists
- Check `rag/embeddings/id_to_chunk.json` is present
- Ensure `sentence-transformers` and `faiss-cpu` are installed
- Check file permissions

### Database Issues
- Ensure `data/` directory exists and is writable
- Check database path in `.env`
- If corrupted, delete `data/smart_environment.db` (will recreate)

### Port Already in Use
- Streamlit default port is 8501
- Change port: `streamlit run app.py --server.port 8502`

## Development Notes

- The project uses a modular architecture with clear separation of concerns
- System prompts are in `llm/prompts.py` for easy modification
- State is persisted in SQLite database (`data/smart_environment.db`)
- RAG embeddings are pre-computed and stored in `rag/embeddings/`
- Configuration can be overridden via environment variables (`.env` file)

## License

[Add your license here]

## Contact

[Add contact information]

