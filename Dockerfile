FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 必要に応じてシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
  python3 python3-pip python3-dev git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Poetryインストール
RUN pip3 install --no-cache-dir poetry

# pyproject.toml & poetry.lock をコピー
COPY pyproject.toml .
# COPY poetry.lock .
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

COPY . /app
RUN mkdir -p /app/artifacts

ENTRYPOINT ["poetry", "run", "python", "train_tcn.py"]
