# 例: Python 3.9 の公式イメージ (CPU用)
FROM python:3.9-slim-bullseye

# 作業ディレクトリを設定
WORKDIR /app

# システム依存のライブラリが必要なら適宜追加
RUN apt-get update && apt-get install -y \
  git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# uvのインストール
RUN pip install uv

# pyproject.toml をコピー
COPY pyproject.toml .

# 依存関係をインストール
RUN uv sync

# ソースコードをコピー (train_tcn.pyなど)
COPY . /app

# 実行コマンドを指定 (エントリポイント)
# ENTRYPOINT ["uv", "run", "python", "run_tcn.py"]