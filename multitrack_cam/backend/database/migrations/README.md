SQLAlchemy `create_all` initializes schema on startup.

For production PostgreSQL migrations, add Alembic revision scripts here and point
`alembic.ini` at `MTMCT_DATABASE_URL`. Keep identity ID allocation transactional
(`id_sequences` row with immediate lock / serializable transaction).
