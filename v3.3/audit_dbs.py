#!/usr/bin/env python3
"""Database completeness audit script."""
import sqlite3
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_FILES = sorted(PROJECT_DIR.glob("*.db"))

print(f"Found {len(DB_FILES)} database files\n")
print("=" * 80)

results = []
for db_path in DB_FILES:
    db_name = db_path.name
    size_mb = db_path.stat().st_size / (1024 * 1024)

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]

        total_rows = 0
        table_details = []
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
            count = cursor.fetchone()[0]
            total_rows += count
            table_details.append(f"{table}={count:,}")

        conn.close()

        status = "[OK]" if total_rows > 0 else "[EMPTY]"
        print(f"{status:8} {db_name:30} {size_mb:8.2f} MB  {len(tables):2} tables  {total_rows:10,} rows")
        if table_details:
            print(f"         Tables: {', '.join(table_details)}")

        results.append({
            'name': db_name,
            'size_mb': size_mb,
            'tables': len(tables),
            'rows': total_rows,
            'table_details': table_details,
            'status': 'ok' if total_rows > 0 else 'empty'
        })

    except Exception as e:
        print(f"[ERROR]  {db_name:30} {size_mb:8.2f} MB  - {e}")
        results.append({
            'name': db_name,
            'size_mb': size_mb,
            'tables': 0,
            'rows': 0,
            'table_details': [],
            'status': f'error: {e}'
        })

    print()

print("=" * 80)
print(f"\nSummary:")
print(f"  Total databases: {len(results)}")
print(f"  OK (has data): {sum(1 for r in results if r['status'] == 'ok')}")
print(f"  Empty: {sum(1 for r in results if r['status'] == 'empty')}")
print(f"  Errors: {sum(1 for r in results if r['status'].startswith('error'))}")
