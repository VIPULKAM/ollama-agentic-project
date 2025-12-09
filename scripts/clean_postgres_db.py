#!/usr/bin/env python3
"""Clean PostgreSQL database.

This script provides options to:
1. Clear all tables (TRUNCATE)
2. Reset specific tables
3. Delete old data
4. Vacuum database
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
    sys.exit(1)


def clean_all_tables():
    """Clear all data from all tables."""
    print("⚠️  WARNING: This will delete ALL data from the database!")
    response = input("Type 'yes' to confirm: ")

    if response.lower() != 'yes':
        print("Cancelled.")
        return False

    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cur = conn.cursor()

        print("\nCleaning database...")

        # Use the truncate_all_tables helper function
        cur.execute("SELECT truncate_all_tables();")
        conn.commit()

        print("✓ All tables truncated")

        # Reset index metadata
        cur.execute("""
            INSERT INTO index_metadata (
                total_chunks,
                total_files,
                total_size_bytes,
                embedding_model,
                last_updated
            ) VALUES (
                0,
                0,
                0,
                'sentence-transformers/all-MiniLM-L6-v2',
                NOW()
            );
        """)
        conn.commit()

        print("✓ Index metadata reset")

        cur.close()
        conn.close()

        print("\n✅ Database cleaned successfully")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def clean_chunks_only():
    """Clear only code chunks (keeps crawl history)."""
    print("This will delete all code chunks but keep crawl history.")
    response = input("Type 'yes' to confirm: ")

    if response.lower() != 'yes':
        print("Cancelled.")
        return False

    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cur = conn.cursor()

        print("\nCleaning chunks...")

        cur.execute("TRUNCATE TABLE code_chunks CASCADE;")
        conn.commit()

        print("✓ Code chunks cleared")

        # Update metadata
        cur.execute("""
            UPDATE index_metadata
            SET total_chunks = 0,
                total_files = 0,
                total_size_bytes = 0,
                last_updated = NOW();
        """)
        conn.commit()

        print("✓ Metadata updated")

        cur.close()
        conn.close()

        print("\n✅ Chunks cleaned successfully")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def vacuum_database():
    """Run VACUUM ANALYZE to optimize database."""
    print("Running VACUUM ANALYZE...")

    try:
        # Need autocommit for VACUUM
        conn = psycopg2.connect(settings.DATABASE_URL)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        cur.execute("VACUUM ANALYZE;")

        print("✓ VACUUM ANALYZE complete")

        cur.close()
        conn.close()

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_menu():
    """Display menu and handle user choice."""
    while True:
        print("\n" + "="*60)
        print("PostgreSQL Database Cleaning")
        print("="*60)
        print("1. Clean all tables (delete everything)")
        print("2. Clean code chunks only (keep crawl history)")
        print("3. Vacuum database (optimize)")
        print("4. Show current stats")
        print("5. Exit")
        print()

        choice = input("Enter choice (1-5): ").strip()

        if choice == '1':
            clean_all_tables()
        elif choice == '2':
            clean_chunks_only()
        elif choice == '3':
            vacuum_database()
        elif choice == '4':
            # Import and run stats script
            from show_postgres_stats import show_stats
            show_stats()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        show_menu()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
