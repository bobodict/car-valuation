import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, inspect, text

from database import ensure_history_metadata_columns
from schemas import HistoryOut


class HistoryMetadataTests(unittest.TestCase):
    def test_sqlite_migration_adds_currency_and_model_version_without_deleting_rows(self):
        engine = create_engine("sqlite://")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE history ("
                    "id INTEGER PRIMARY KEY, model VARCHAR(255), price FLOAT)"
                )
            )
            connection.execute(
                text("INSERT INTO history (id, model, price) VALUES (1, 'Amaze', 505000)")
            )

        ensure_history_metadata_columns(engine)

        columns = {column["name"] for column in inspect(engine).get_columns("history")}
        with engine.connect() as connection:
            row = connection.execute(text("SELECT model, price FROM history WHERE id = 1")).one()

        self.assertIn("currency", columns)
        self.assertIn("model_version", columns)
        self.assertEqual(tuple(row), ("Amaze", 505000.0))

    def test_mysql_migration_adds_missing_columns_without_dropping_history(self):
        executed = []

        class Connection:
            def exec_driver_sql(self, statement):
                executed.append(statement)

        class Engine:
            dialect = SimpleNamespace(
                name="mysql",
                identifier_preparer=SimpleNamespace(quote=lambda name: f"`{name}`"),
            )

            def begin(self):
                class Context:
                    def __enter__(self):
                        return Connection()

                    def __exit__(self, *args):
                        return False

                return Context()

        inspector = SimpleNamespace(
            has_table=lambda table_name: True,
            get_columns=lambda table_name: [{"name": "id"}, {"name": "price"}],
        )
        with patch("database.inspect", return_value=inspector):
            ensure_history_metadata_columns(Engine())

        self.assertEqual(len(executed), 2)
        self.assertIn("ALTER TABLE `history` ADD COLUMN `currency` VARCHAR(16) NULL", executed)
        self.assertIn("ALTER TABLE `history` ADD COLUMN `model_version` VARCHAR(64) NULL", executed)

    def test_history_response_backfills_metadata_for_legacy_rows(self):
        payload = {
            "id": 1,
            "model": "Amaze",
            "city": "Pune",
            "mileage": 87150,
            "year": 2017,
            "month": 6,
            "gearbox": "Manual",
            "emission": "unknown",
            "price": 505000,
            "currency": None,
            "model_version": None,
            "created_at": "2026-07-22T12:00:00",
            "status": "experimental",
        }

        result = HistoryOut.model_validate(payload)

        self.assertEqual(result.currency, "INR")
        self.assertEqual(result.model_version, "unknown")


if __name__ == "__main__":
    unittest.main()
