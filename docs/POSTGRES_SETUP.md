# PostgreSQL Local Development Setup

This guide helps team members set up a local PostgreSQL database with pgvector for the AI Coding Agent project.

## Prerequisites

- Docker and Docker Compose installed
- Git repository cloned
- Terminal access

## Quick Start (5 minutes)

### 1. Start PostgreSQL Database

```bash
# Start PostgreSQL only
docker-compose up -d postgres

# Or start with pgAdmin (database management UI)
docker-compose --profile with-pgadmin up -d
```

### 2. Verify Database is Running

```bash
# Check container status
docker-compose ps

# Check database health
docker-compose exec postgres pg_isready -U ai_agent_user -d ai_agent

# Expected output: postgres:5432 - accepting connections
```

### 3. Configure Application

Copy and update `.env` file:

```bash
# Copy example env file if you haven't already
cp .env.example .env

# Edit .env and set PostgreSQL configuration:
# ENABLE_POSTGRES_STORAGE=true
# DATABASE_URL=postgresql://ai_agent_user:dev_password_change_in_production@localhost:5432/ai_agent
```

### 4. Test Connection

```bash
# Run the database test script
python scripts/test_postgres_connection.py

# Expected output: ✓ PostgreSQL connection successful
```

## Database Details

### Connection Information

- **Host**: localhost
- **Port**: 5432
- **Database**: ai_agent
- **Username**: ai_agent_user
- **Password**: dev_password_change_in_production
- **Connection String**: `postgresql://ai_agent_user:dev_password_change_in_production@localhost:5432/ai_agent`

### Database Schema

The database includes:
- **code_chunks**: Stores code chunks with embeddings (384-dim vectors)
- **crawled_urls**: Tracks crawled URLs for deduplication
- **index_metadata**: Stores indexing metadata and statistics
- **crawl_sessions**: Tracks crawling sessions

### Vector Search

The database uses **pgvector** extension with:
- HNSW indexes for fast similarity search
- L2 distance for vector similarity (matches FAISS behavior)
- Optimized for 384-dimension embeddings (sentence-transformers/all-MiniLM-L6-v2)

## pgAdmin Access (Optional)

If you started with `--profile with-pgadmin`:

1. Open browser: http://localhost:5050
2. Login credentials:
   - Email: admin@aiagent.local
   - Password: admin
3. Server connection is pre-configured as "AI Agent Local"

## Common Operations

### View Database Statistics

```bash
# Connect to database
docker-compose exec postgres psql -U ai_agent_user -d ai_agent

# Run queries
SELECT * FROM v_index_statistics;
SELECT * FROM v_crawl_statistics;

# Exit psql
\q
```

### Reset Database

```bash
# Stop containers
docker-compose down

# Remove volumes (WARNING: Deletes all data!)
docker volume rm ai-agent-postgres-data

# Restart fresh
docker-compose up -d postgres
```

### Backup Database

```bash
# Create backup
docker-compose exec postgres pg_dump -U ai_agent_user ai_agent > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
docker-compose exec -T postgres psql -U ai_agent_user ai_agent < backup_20241208_120000.sql
```

### View Logs

```bash
# View PostgreSQL logs
docker-compose logs postgres

# Follow logs in real-time
docker-compose logs -f postgres
```

### Connect with psql

```bash
# Interactive shell
docker-compose exec postgres psql -U ai_agent_user -d ai_agent

# Run single query
docker-compose exec postgres psql -U ai_agent_user -d ai_agent -c "SELECT COUNT(*) FROM code_chunks;"
```

## Utility Scripts

The project includes helper scripts in `scripts/` directory:

### Test Connection
```bash
python scripts/test_postgres_connection.py
```

### View Statistics
```bash
python scripts/show_postgres_stats.py
```

### Clean Database
```bash
python scripts/clean_postgres_db.py
```

## Troubleshooting

### Port 5432 Already in Use

If you have PostgreSQL installed locally:

```bash
# Option 1: Stop local PostgreSQL
sudo systemctl stop postgresql

# Option 2: Change Docker port in docker-compose.yml
# Change "5432:5432" to "5433:5432"
# Then use: postgresql://ai_agent_user:...@localhost:5433/ai_agent
```

### Permission Denied on Init Scripts

```bash
# Make scripts executable
chmod +r scripts/init_postgres_schema.sql
chmod +r scripts/seed_data.sql
chmod +r scripts/pgadmin_servers.json
```

### Database Not Initializing

```bash
# Check initialization logs
docker-compose logs postgres | grep -A 20 "database system is ready"

# If issues persist, reset completely
docker-compose down -v
docker-compose up -d postgres
```

### Connection Refused

```bash
# Wait for database to be ready (takes 10-30 seconds on first start)
docker-compose exec postgres pg_isready -U ai_agent_user -d ai_agent

# Check if container is running
docker-compose ps postgres

# Check container logs for errors
docker-compose logs postgres
```

## Development Workflow

### Typical Development Session

```bash
# 1. Start database
docker-compose up -d postgres

# 2. Verify connection
python scripts/test_postgres_connection.py

# 3. Run your code/tests
pytest tests/test_rag/test_postgres_backend.py -v

# 4. View statistics
python scripts/show_postgres_stats.py

# 5. Stop when done (optional - can leave running)
docker-compose stop postgres
```

### Working with Team Members

The Docker setup ensures consistency across team members:
- Same PostgreSQL version (16)
- Same pgvector extension
- Same schema and configuration
- Isolated from system PostgreSQL

Just share:
1. `.env` configuration (without sensitive data)
2. Any schema migrations in `scripts/`
3. Docker Compose updates

## Environment Variables

Required in `.env` file:

```bash
# Enable PostgreSQL mode
ENABLE_POSTGRES_STORAGE=true

# Database connection
DATABASE_URL=postgresql://ai_agent_user:dev_password_change_in_production@localhost:5432/ai_agent

# Connection pool settings (optional)
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_CONNECTION_TIMEOUT=30

# pgvector settings (optional)
PGVECTOR_INDEX_TYPE=hnsw
PGVECTOR_HNSW_M=16
PGVECTOR_HNSW_EF_CONSTRUCTION=64
```

## Security Notes

### Development vs Production

The default configuration is for **LOCAL DEVELOPMENT ONLY**:
- ⚠️ Weak password in docker-compose.yml
- ⚠️ Database exposed on localhost:5432
- ⚠️ pgAdmin with default credentials

### For Production

1. Use strong passwords
2. Use environment variables for secrets
3. Enable SSL/TLS
4. Restrict network access
5. Use managed PostgreSQL service (AWS RDS, Google Cloud SQL, etc.)

## Next Steps

1. ✅ Database is running
2. ✅ Connection verified
3. ✅ Schema initialized
4. ⏭️ Run `pytest tests/test_rag/test_postgres_backend.py` to verify backend
5. ⏭️ Try indexing: `python -c "from src.rag.indexer import build_index; build_index()"`
6. ⏭️ Try search: `python -c "from src.rag.retriever import search; print(search('fastapi'))"`

## Resources

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/16/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- Project Architecture: `docs/ARCHITECTURE.md`
- PostgreSQL Backend Code: `src/rag/postgres_backend.py`

## Support

Issues with PostgreSQL setup?

1. Check logs: `docker-compose logs postgres`
2. Verify health: `docker-compose exec postgres pg_isready -U ai_agent_user -d ai_agent`
3. Reset database: `docker-compose down -v && docker-compose up -d postgres`
4. Ask team members or check project documentation
