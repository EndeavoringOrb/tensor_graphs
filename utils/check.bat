@echo off
call .\.venv\Scripts\activate
ruff format
ruff check --fix
pyright
pytest