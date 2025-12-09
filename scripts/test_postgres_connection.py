#!/usr/bin/env python3
"""Test PostgreSQL database connection.

This script verifies that:
1. PostgreSQL server is accessible
2. Credentials are correct
3. Database schema is initialized
4. pgvector extension is available
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psycopg2
    from src.config.settings import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've installed dependencies: pip install -r requirements.txt")
    sys.exit(1)


def test_connection():
    """Test database connection and verify setup."""
    print("Testing PostgreSQL connection...")
    print(f"Database URL: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'N/A'}")
    print()

    try:
        # Connect to database
        conn = psycopg2.connect(settings.DATABASE_URL)
        cur = conn.cursor()

        # Test 1: Basic connection
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print("✓ Connection successful")
        print(f"  PostgreSQL version: {version.split(',')[0]}")

        # Test 2: Check pgvector extension
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cur.fetchone()
        if vector_ext:
            print("✓ pgvector extension is installed")
        else:
            print("❌ pgvector extension not found")
            print("  Run: CREATE EXTENSION vector;")

        # Test 3: Check tables exist
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]

        expected_tables = ['code_chunks', 'crawled_urls', 'index_metadata', 'crawl_sessions']

        print("\n✓ Tables found:")
        for table in tables:
            status = "✓" if table in expected_tables else "?"
            print(f"  {status} {table}")

        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            print(f"\n⚠️  Missing tables: {', '.join(missing_tables)}")
            print("  Run initialization script: scripts/init_postgres_schema.sql")

        # Test 4: Check index statistics
        cur.execute("SELECT * FROM index_metadata LIMIT 1;")
        metadata = cur.fetchone()
        if metadata:
            print("\n✓ index_metadata table is initialized")
        else:
            print("\n⚠️  index_metadata table is empty (this is OK for fresh install)")

        # Test 5: Check views
        cur.execute("""
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        views = [row[0] for row in cur.fetchall()]

        if views:
            print("\n✓ Views found:")
            for view in views:
                print(f"  ✓ {view}")
        else:
            print("\n⚠️  No views found (run initialization script)")

        # Test 6: Test vector operations
        try:
            cur.execute("SELECT '[1,2,3]'::vector;")
            print("\n✓ Vector type operations working")
        except Exception as e:
            print(f"\n❌ Vector operations failed: {e}")

        # Close connection
        cur.close()
        conn.close()

        print("\n" + "="*50)
        print("✅ PostgreSQL setup is complete and working!")
        print("="*50)
        print("\nYou can now:")
        print("1. Run the application with ENABLE_POSTGRES_STORAGE=true")
        print("2. Run tests: pytest tests/test_rag/test_postgres_backend.py")
        print("3. View stats: python scripts/show_postgres_stats.py")

        return True

    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Is PostgreSQL running? docker-compose ps")
        print("2. Is DATABASE_URL correct in .env?")
        print("3. Wait 10-30 seconds after starting container")
        print("4. Check logs: docker-compose logs postgres")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
