"""
List all users in the database so you can identify which IDs correspond to
User 1 (non-vegetarian) and User 2 (vegetarian).

Update USER1_ID / USER2_ID in src/config.py after reviewing the output.

Usage:
    python scripts/list_users.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database.connection import DBConnection
from src.database.loader import list_users
import src.config as cfg


def main():
    db = DBConnection(cfg.DB_HOST, cfg.DB_PORT, cfg.DB_USER, cfg.DB_PASSWORD, cfg.DB_NAME)

    if not db.test():
        print("ERROR: Cannot connect to database. Check credentials in src/config.py")
        sys.exit(1)

    users = list_users(db)
    print(f"\n{'ID':>5}  {'Name':<20} {'Surname':<15} {'Username':<25} {'Age':>4} {'Gender':<10}")
    print("-" * 85)
    for u in users:
        print(
            f"{u['id']:>5}  {str(u['name'] or ''):<20} {str(u['surname'] or ''):<15} "
            f"{str(u['username']):<25} {str(u['age'] or '?'):>4} {str(u['gender'] or '?'):<10}"
        )
    print(f"\nTotal users: {len(users)}")
    print("\nUpdate USER1_ID and USER2_ID in src/config.py with the correct IDs.")


if __name__ == "__main__":
    main()
