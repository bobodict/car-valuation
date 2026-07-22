import unittest

from sqlalchemy import create_engine, inspect, text

from database import ensure_history_metadata_columns


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


if __name__ == "__main__":
    unittest.main()
