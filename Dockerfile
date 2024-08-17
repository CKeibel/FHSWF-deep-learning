FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry install --no-root || \
    (pip3 install torch==2.0.0 -f https://download.pytorch.org/whl/cu113 && poetry install --no-root)

COPY . /app

EXPOSE 7860

CMD ["poetry", "run", "python", "src/multimodal_rag/__main__.py"]