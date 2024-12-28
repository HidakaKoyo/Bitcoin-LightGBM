# 例: Python 3.9 の公式イメージ (CPU用)
FROM python:3.9-slim-bullseye

# 作業ディレクトリを設定
WORKDIR /app

# システム依存のライブラリが必要なら適宜追加
RUN apt-get update && apt-get install -y \
  git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Poetryのインストール
# (Poetry公式のインストールスクリプトを使う方法でもOK)
RUN pip install uv

# pyproject.toml & poetry.lock をコピー
COPY pyproject.toml .
# すでにpoetry.lockがある場合はコピー
# COPY poetry.lock .

# 依存関係をインストール
RUN uv sync

# ソースコードをコピー (train_tcn.pyなど)
COPY . /app

# artifactsディレクトリを作成 (必要なら)
RUN mkdir -p /app/artifacts

# 実行コマンドを指定 (エントリポイント)
ENTRYPOINT ["poetry", "run", "python", "train_tcn.py"]
