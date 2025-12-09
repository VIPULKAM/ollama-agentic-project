#!/bin/bash
# Quick setup script for PostgreSQL local development

set -e  # Exit on error

echo "==========================================="
echo "PostgreSQL Setup for AI Coding Agent"
echo "==========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker is installed"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  Please update .env with PostgreSQL settings:"
    echo "   ENABLE_POSTGRES_STORAGE=true"
    echo "   DATABASE_URL=postgresql://ai_agent_user:dev_password_change_in_production@localhost:5432/ai_agent"
    echo ""
fi

# Check if containers are already running
if docker-compose ps postgres | grep -q "Up"; then
    echo "⚠️  PostgreSQL container is already running"
    echo ""
    read -p "Do you want to restart it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Restarting PostgreSQL..."
        docker-compose restart postgres
    fi
else
    # Start PostgreSQL
    echo "Starting PostgreSQL container..."
    docker-compose up -d postgres
    echo ""
fi

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U ai_agent_user -d ai_agent &> /dev/null; then
        echo "✓ PostgreSQL is ready!"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Test connection
echo ""
echo "Testing database connection..."
python3 scripts/test_postgres_connection.py

echo ""
echo "==========================================="
echo "✅ Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Update .env file with PostgreSQL settings (if needed)"
echo "2. Run the application: python main.py"
echo "3. View stats: python scripts/show_postgres_stats.py"
echo ""
echo "Optional:"
echo "- Start pgAdmin: docker-compose --profile with-pgadmin up -d"
echo "- Access pgAdmin at: http://localhost:5050"
echo "  (Login: admin@aiagent.local / admin)"
echo ""
