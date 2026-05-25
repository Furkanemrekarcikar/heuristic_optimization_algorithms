import mysql.connector
from contextlib import contextmanager
from typing import Iterator


class DBConnection:
    """Thin MySQL connection wrapper with context-manager cursor support."""

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self._cfg = {
            "host":      host,
            "port":      port,
            "user":      user,
            "password":  password,
            "database":  database,
            "charset":   "utf8mb4",
            "use_unicode": True,
        }

    def connect(self) -> mysql.connector.MySQLConnection:
        return mysql.connector.connect(**self._cfg)

    @contextmanager
    def cursor(self, dictionary: bool = True) -> Iterator:
        conn = self.connect()
        try:
            cur = conn.cursor(dictionary=dictionary)
            yield cur
        finally:
            cur.close()
            conn.close()

    def test(self) -> bool:
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception as exc:
            print(f"[DBConnection] test failed: {exc}")
            return False
