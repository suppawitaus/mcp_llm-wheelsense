# MCP Smart Environment System

A Streamlit-based smart environment assistant for elderly/disabled users. The system uses LLM (via Ollama) to control devices, manage schedules, and provide health-related guidance through RAG (Retrieval-Augmented Generation).

## Features

- **Device Control**: Control lights, AC, TV, Fan, and Alarm across multiple rooms
- **Schedule Management**: Add, modify, and delete daily schedule items
- **Health Knowledge Base**: RAG system provides tailored health recommendations based on user conditions
- **Notification System**: Proactive house checks and activity reminders
- **Natural Language Interface**: Chat-based interaction powered by LLM

## Prerequisites

**Docker and Docker Compose are required.** No local Python installation needed.

- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** 2.0+ (included with Docker Desktop)

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd mcp_llm

# Copy environment file (optional - defaults work)
cp .env.example .env

# Start all services
docker compose up --build
```

The application will be available at `http://localhost:8501`

**First Run:** The required model (`qwen2.5:7b`) will be automatically downloaded during startup. This may take a few minutes depending on your internet connection. The model is persisted in a Docker volume, so it only needs to be downloaded once.

## Architecture

The system runs in Docker with the following services:

```
┌─────────────────┐
│   Streamlit     │  Port 8501
│   Application   │  ──────────┐
└────────┬────────┘              │
         │                       │
         │ HTTP                  │ User
         │                       │ Browser
┌────────▼────────┐              │
│     Ollama      │  Port 11434  │
│   LLM Service   │              │
└─────────────────┘              │
                                  │
┌─────────────────┐              │
│  SQLite DB      │  Volume      │
│  (data/)        │  Mount       │
└─────────────────┘              │
                                  │
┌─────────────────┐              │
│  RAG Embeddings │  Volume      │
│  (rag/)         │  Mount       │
└─────────────────┘              │
```

### Services

1. **app** (Streamlit Application)
   - Python application serving the web UI
   - Connects to Ollama service for LLM processing
   - Uses SQLite database for state persistence
   - Accesses RAG embeddings for health knowledge

2. **ollama** (LLM Service)
   - Runs Ollama server for language model inference
   - Models are persisted in Docker volume `ollama_data`

3. **ollama-init** (Model Initialization)
   - Automatically checks for and downloads the required model on first run
   - Runs once after Ollama service is healthy
   - Skips download if model already exists in the volume

## Configuration

Configuration is done via environment variables in `.env` file.

### Required Variables

- `OLLAMA_HOST`: Ollama service URL (default: `http://ollama:11434` in Docker)
- `MODEL_NAME`: LLM model name (default: `qwen2.5:7b`)

### Optional Variables

- `DATABASE_PATH`: SQLite database path (default: `data/smart_environment.db`)
- `DATABASE_BACKUP_DIR`: Backup directory (default: `data/backups`)
- `ENABLE_DATABASE_LOGGING`: Enable SQL logging (default: `false`)
- `USE_COMPACT_PROMPT`: Use compact prompts (default: `false`)
- `STREAMLIT_SERVER_PORT`: Streamlit port (default: `8501`)
- `STREAMLIT_SERVER_ADDRESS`: Streamlit address (default: `0.0.0.0`)

See `.env.example` for all available options.

## Usage

1. **Start the system:**
   ```bash
   docker compose up --build
   ```

2. **Access the application:**
   - Open `http://localhost:8501` in your browser

3. **Interact via chat:**
   - Ask questions: "What devices are on?"
   - Control devices: "Turn on the light"
   - Manage schedule: "I have a meeting at 14:00"
   - Get health advice: "What should I eat for breakfast?" (if user has health conditions)

4. **Room Map**: Click on rooms in the left panel to change user location

5. **Schedule**: View and manage daily schedule items

## Docker Commands

### Start Services
```bash
docker compose up --build
```

### Start in Background
```bash
docker compose up -d --build
```

### View Logs
```bash
docker compose logs -f
```

### Stop Services
```bash
docker compose down
```

### Stop and Remove Volumes
```bash
docker compose down -v
```

### Rebuild After Code Changes
```bash
docker compose up --build
```

### Access Application Container
```bash
docker compose exec app bash
```

### Access Ollama Container
```bash
docker compose exec ollama bash
```

## Data Persistence

- **Database**: Stored in `data/` directory (mounted as volume)
- **Ollama Models**: Stored in Docker volume `ollama_data`
- **RAG Embeddings**: Included in image, read-only mount

To reset the database:
```bash
docker compose down
rm -rf data/*.db
docker compose up --build
```

## Troubleshooting

### Application Won't Start

**Check service status:**
```bash
docker compose ps
```

**View logs:**
```bash
docker compose logs app
docker compose logs ollama
```

### Ollama Connection Issues

**Verify Ollama is running:**
```bash
docker compose ps ollama
```

**Check Ollama health:**
```bash
docker compose exec ollama curl http://localhost:11434/api/tags
```

**Verify model is available:**
```bash
docker compose exec ollama ollama list
```

**Check model initialization logs:**
```bash
docker compose logs ollama-init
```

### Port Already in Use

If port 8501 is already in use, change it in `.env`:
```env
STREAMLIT_SERVER_PORT=8502
```

Then update `docker-compose.yml` port mapping:
```yaml
ports:
  - "8502:8501"
```

### RAG System Not Working

RAG embeddings are included in the repository. If issues occur:
- Verify `rag/embeddings/faiss_index.bin` exists
- Verify `rag/embeddings/id_to_chunk.json` exists
- Check application logs: `docker compose logs app | grep RAG`

### Database Issues

**Reset database:**
```bash
docker compose down
rm -rf data/*.db data/backups/*
docker compose up --build
```

**Check database file:**
```bash
ls -lh data/*.db
```

### Build Issues

**Clear Docker cache and rebuild:**
```bash
docker compose build --no-cache
docker compose up
```

**Check Docker resources:**
```bash
docker system df
docker system prune  # Remove unused resources
```

## Development

### Project Structure

```
mcp_llm/
├── Dockerfile              # Application container definition
├── docker-compose.yml      # Service orchestration
├── .dockerignore           # Build exclusions
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit UI entry point
├── config.py               # Configuration constants
├── core/                   # Core business logic
├── database/               # Database models and manager
├── llm/                    # LLM interaction
├── mcp/                    # MCP protocol components
├── rag/                    # RAG system
│   ├── embeddings/        # Pre-computed embeddings
│   └── retrieval/         # Retrieval logic
├── services/               # Application services
└── utils/                  # Utilities
```

### Making Code Changes

1. Edit code files
2. Rebuild and restart:
   ```bash
   docker compose up --build
   ```

### Accessing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f app
docker compose logs -f ollama
```

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

## License

[Add your license here]

## Contact

[Add contact information]
