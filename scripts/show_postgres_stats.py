#!/usr/bin/env python3
"""Show PostgreSQL database statistics.

Displays:
- Index statistics (chunks, files, size)
- Crawl statistics (URLs, sessions)
- Table sizes
- Recent activity
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from src.config.settings import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def show_stats():
    """Display database statistics."""
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        print("="*60)
        print("PostgreSQL Database Statistics")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Index Statistics
        print("📊 Index Statistics")
        print("-" * 60)
        cur.execute("SELECT * FROM v_index_statistics;")
        index_stats = cur.fetchone()

        if index_stats:
            print(f"  Total Chunks:      {index_stats['total_chunks']:,}")
            print(f"  Total Files:       {index_stats['total_files']:,}")
            print(f"  Total Size:        {index_stats['total_size_mb']:.2f} MB")
            print(f"  Avg Chunk Size:    {index_stats['avg_chunk_size']:.0f} chars")
            print(f"  Embedding Model:   {index_stats['embedding_model']}")
            print(f"  Last Updated:      {index_stats['last_updated']}")
        else:
            print("  No index data yet")

        print()

        # Crawl Statistics
        print("🌐 Crawl Statistics")
        print("-" * 60)
        cur.execute("SELECT * FROM v_crawl_statistics;")
        crawl_stats = cur.fetchone()

        if crawl_stats:
            print(f"  Total URLs:        {crawl_stats['total_urls']:,}")
            print(f"  Successfully Crawled: {crawl_stats['successful_crawls']:,}")
            print(f"  Failed Crawls:     {crawl_stats['failed_crawls']:,}")
            print(f"  Total Sessions:    {crawl_stats['total_sessions']:,}")
            print(f"  Last Crawl:        {crawl_stats['last_crawl_time']}")
        else:
            print("  No crawl data yet")

        print()

        # Table Sizes
        print("💾 Table Sizes")
        print("-" * 60)
        cur.execute("""
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY size_bytes DESC;
        """)
        tables = cur.fetchall()

        for table in tables:
            print(f"  {table['tablename']:20s} {table['size']:>10s}")

        print()

        # Recent Activity
        print("📅 Recent Activity")
        print("-" * 60)
        cur.execute("""
            SELECT
                file_path,
                chunk_type,
                LENGTH(content) as content_length,
                created_at
            FROM code_chunks
            ORDER BY created_at DESC
            LIMIT 5;
        """)
        recent_chunks = cur.fetchall()

        if recent_chunks:
            print("  Recent chunks:")
            for chunk in recent_chunks:
                print(f"    {chunk['created_at']} | {chunk['file_path'][:40]:40s} | {chunk['chunk_type']:10s} | {chunk['content_length']} chars")
        else:
            print("  No chunks indexed yet")

        print()

        cur.execute("""
            SELECT
                url,
                crawl_status,
                crawled_at
            FROM crawled_urls
            ORDER BY crawled_at DESC
            LIMIT 5;
        """)
        recent_urls = cur.fetchall()

        if recent_urls:
            print("  Recent crawls:")
            for url_rec in recent_urls:
                status_icon = "✓" if url_rec['crawl_status'] == 'success' else "✗"
                print(f"    {status_icon} {url_rec['crawled_at']} | {url_rec['url'][:50]}")
        else:
            print("  No URLs crawled yet")

        print()

        # Connection Info
        print("🔗 Connection Info")
        print("-" * 60)
        cur.execute("SELECT current_database(), current_user;")
        db_info = cur.fetchone()
        print(f"  Database:          {db_info['current_database']}")
        print(f"  User:              {db_info['current_user']}")

        cur.execute("SHOW server_version;")
        version = cur.fetchone()['server_version']
        print(f"  PostgreSQL:        {version}")

        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cur.fetchone()
        if vector_ext:
            print(f"  pgvector:          ✓ Installed (v{vector_ext['extversion']})")
        else:
            print(f"  pgvector:          ✗ Not installed")

        print()
        print("="*60)

        cur.close()
        conn.close()

        return True

    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\nIs PostgreSQL running? Try: docker-compose ps")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = show_stats()
    sys.exit(0 if success else 1)
