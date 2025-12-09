# PostgreSQL Quick Start for Team Members

**⏱️ Setup time: 5 minutes**

This guide helps you quickly set up a local PostgreSQL database for the AI Coding Agent project.

## One-Line Setup (Automated)

```bash
./scripts/setup_postgres.sh
```

That's it! The script will:
- ✅ Start PostgreSQL with Docker
- ✅ Initialize the database schema
- ✅ Verify the connection
- ✅ Show next steps

## Manual Setup (3 Steps)

### Step 1: Start PostgreSQL

```bash
docker-compose up -d postgres
```

### Step 2: Verify Connection

```bash
python scripts/test_postgres_connection.py
```

Expected output:
```
✓ Connection successful
✓ pgvector extension is installed
✓ Tables found: code_chunks, crawled_urls, index_metadata, crawl_sessions
✅ PostgreSQL setup is complete and working!
```

### Step 3: Configure Application

Edit `.env` file (create from `.env.example` if needed):

```bash
# Enable PostgreSQL mode
ENABLE_POSTGRES_STORAGE=true

# Database connection (local Docker)
DATABASE_URL=postgresql://ai_agent_user:dev_password_change_in_production@localhost:5432/ai_agent
```

## Verify Everything Works

```bash
# View database statistics
python scripts/show_postgres_stats.py

# Run backend tests (if available)
pytest tests/test_rag/test_postgres_backend.py -v
```

## Common Commands

```bash
# Start database
docker-compose up -d postgres

# Stop database
docker-compose stop postgres

# View logs
docker-compose logs postgres

# Clean database (WARNING: deletes all data)
python scripts/clean_postgres_db.py

# Access database directly
docker-compose exec postgres psql -U ai_agent_user -d ai_agent
```

## Optional: pgAdmin (Database UI)

```bash
# Start PostgreSQL + pgAdmin
docker-compose --profile with-pgadmin up -d

# Access pgAdmin at: http://localhost:5050
# Login: admin@aiagent.local / admin
# Server "AI Agent Local" is pre-configured
```

## Troubleshooting

### "Connection refused"
Wait 10-30 seconds after starting container, then try again.

### "Port 5432 already in use"
You have local PostgreSQL running:
```bash
sudo systemctl stop postgresql
# Or change port in docker-compose.yml
```

### "Permission denied"
Make scripts executable:
```bash
chmod +x scripts/*.sh scripts/*.py
```

### Database not initializing
```bash
# Reset completely
docker-compose down -v
docker-compose up -d postgres
```

## What Gets Created

- **Docker Container**: `ai-agent-postgres` running PostgreSQL 16 + pgvector
- **Database**: `ai_agent`
- **Tables**:
  - `code_chunks` - Code chunks with embeddings
  - `crawled_urls` - URL tracking
  - `index_metadata` - Index statistics
  - `crawl_sessions` - Crawl session tracking
- **Views**: `v_index_statistics`, `v_crawl_statistics`

## Connection Details

| Setting | Value |
|---------|-------|
| Host | localhost |
| Port | 5432 |
| Database | ai_agent |
| Username | ai_agent_user |
| Password | dev_password_change_in_production |

⚠️ **Note**: These are development credentials. Never use in production!

## Full Documentation

For detailed documentation, see:
- 📖 **Setup Guide**: [`docs/POSTGRES_SETUP.md`](docs/POSTGRES_SETUP.md)
- 🏗️ **Architecture**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 📝 **Schema**: [`scripts/init_postgres_schema.sql`](scripts/init_postgres_schema.sql)

## Need Help?

1. Check logs: `docker-compose logs postgres`
2. Verify health: `docker-compose exec postgres pg_isready -U ai_agent_user -d ai_agent`
3. Full documentation: [`docs/POSTGRES_SETUP.md`](docs/POSTGRES_SETUP.md)
4. Ask team members!

---

**Ready to code?** You're all set! 🚀

The database is running at `localhost:5432` and ready to use.
