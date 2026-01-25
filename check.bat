@echo off
call .\.venv\Scripts\activate
pyright
pytest
black .
python check.py