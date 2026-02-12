import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")


def now():
    return datetime.utcnow().isoformat()


def main():
    # Recreate a clean DB (good for demos/assignments).
    # If you want to keep old users/history, comment out the next 2 lines.
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS plants(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            is_toxic INTEGER NOT NULL CHECK(is_toxic IN (0,1)),
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            query TEXT NOT NULL,
            method TEXT NOT NULL,
            result_label TEXT NOT NULL,
            confidence INTEGER,
            image_filename TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    # Toxicity is for pets/cats.
    plants = [
        ("Aloe Vera", 0),   # toxic to cats/dogs
        ("Lily", 1),        # very toxic (especially to cats)
        ("Monstera", 1),    # irritant/toxic to pets
    ]

    cur.executemany(
        "INSERT OR REPLACE INTO plants(name,is_toxic,created_at) VALUES(?,?,?)",
        [(n, t, now()) for n, t in plants],
    )

    con.commit()
    con.close()
    print(f"Database created: {DB_PATH}")


if __name__ == "__main__":
    main()
