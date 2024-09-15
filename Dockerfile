FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
COPY . .

RUN poetry install --no-root

EXPOSE 7860

CMD ["poetry", "run", "python", "main.py"]
