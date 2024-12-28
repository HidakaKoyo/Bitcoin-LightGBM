import os
import time
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================
#  ログ設定
# =========================================
def setup_logger(log_file="training.log"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # すでにハンドラがある場合はスキップ
    if not logger.handlers:
        # フォーマッタ
        formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

        # ファイル出力用ハンドラ
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # コンソール出力用ハンドラ(オプション)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# =========================================
#  TCNモデル定義
# =========================================
class Chomp1d(nn.Module):
    """TCNの因果性(未来情報を使わない)を保つために余分なタイムステップをカットする"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super().__init__()
        # 最初のConv
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 2つ目のConv
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 残差接続のための1x1Conv（チャネル数が合わない場合に対応）
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # 残差接続
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        num_inputs: 入力チャネル数
        num_channels: [c1, c2, ...] 各層のチャネル数
        kernel_size: 畳み込みカーネルサイズ
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch, channels, seq_len)
        """
        return self.network(x)


class TCNRegressor(nn.Module):
    def __init__(self, input_size, tcn_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        # TCN本体
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        # TCN出力 -> 全結合
        # TCN出力のチャネル数 = tcn_channels[-1]、出力は1次元(Close価格の1ステップ予測)
        self.fc = nn.Linear(tcn_channels[-1], 1)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size)
        """
        # TCNは (batch, channel, seq_len) を期待
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        tcn_out = self.tcn(x)  # (batch, tcn_channels[-1], seq_len)
        # 最後のタイムステップのみ取り出して全結合へ
        last_step = tcn_out[:, :, -1]  # (batch, tcn_channels[-1])
        out = self.fc(last_step)  # (batch, 1)
        return out


# =========================================
#  データ読み込み＆前処理
# =========================================
def load_data(csv_path="./data/btcusd_1-min_data.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # このサンプルではClose価格の1ステップ予測を行う
    # 必要に応じてOpen, High, Low, Volumeなど追加特徴量にしてください
    df = df.dropna(subset=["Close"])
    close_vals = df["Close"].values.astype(np.float32)

    return close_vals


def create_sequences(data, seq_len):
    """
    data: (N,) の1次元配列
    seq_len: 入力に使う過去ステップ数
    戻り値: X, y
      X shape: (サンプル数, seq_len, 1)  [特徴量次元=1とする]
      y shape: (サンプル数,)
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x = data[i : i + seq_len]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    X = np.array(xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y


# =========================================
#  メイン処理
# =========================================
def main():
    logger = setup_logger(log_file="training.log")

    # 全体実行時間の測定開始
    start_time_overall = time.time()

    # GPUが使用可能か判定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---------------------------------------
    # データ読み込み
    # ---------------------------------------
    data = load_data(csv_path="./data/btcusd_1-min_data.csv")
    logger.info(f"Data loaded. Total samples: {len(data)}")

    # スケーリング (推奨)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # シーケンス変換
    seq_len = 30  # 過去30分のデータを使って次の1分を予測
    X, y = create_sequences(data_scaled, seq_len)
    logger.info(f"Sequence dataset created. X.shape={X.shape}, y.shape={y.shape}")

    # データ分割
    test_ratio = 0.1
    test_size = int(len(X) * test_ratio)
    train_size = len(X) - test_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logger.info(f"Train size: {train_size}, Test size: {test_size}")

    # DataLoader作成
    class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # ---------------------------------------
    # モデル構築
    # ---------------------------------------
    tcn_channels = [32, 32, 32]  # 必要に応じて調整
    model = TCNRegressor(
        input_size=1, tcn_channels=tcn_channels, kernel_size=3, dropout=0.2
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    logger.info("Starting training...")

    # ========================================
    #  GPU上での演算時間計測
    # ========================================
    start_gpu_event = torch.cuda.Event(enable_timing=True)
    end_gpu_event = torch.cuda.Event(enable_timing=True)

    # 訓練開始時にGPUイベントを記録
    if device.type == "cuda":
        start_gpu_event.record()

    # ---------------------------------------
    # 学習ループ
    # ---------------------------------------
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)  # shape (batch, 1)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.6f}")

    # 訓練終了時にGPUイベントを記録
    if device.type == "cuda":
        end_gpu_event.record()
        # GPUの演算完了を待機(synchronize)
        torch.cuda.synchronize()
        # イベント間の経過時間 (ミリ秒->秒)
        gpu_time = start_gpu_event.elapsed_time(end_gpu_event) / 1000.0
        logger.info(f"Total GPU time (training loop): {gpu_time:.2f} sec")

    # ---------------------------------------
    # テスト推論
    # ---------------------------------------
    logger.info("Starting inference...")
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch).squeeze()  # shape: (batch,)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # スケーリングを元に戻す
    predictions_unscaled = scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).flatten()
    actuals_unscaled = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # 評価指標
    rmse = np.sqrt(np.mean((predictions_unscaled - actuals_unscaled) ** 2))
    logger.info(f"Test RMSE: {rmse:.4f}")

    # ---------------------------------------
    # 全体実行時間
    # ---------------------------------------
    end_time_overall = time.time()
    total_time = end_time_overall - start_time_overall
    logger.info(f"Overall execution time: {total_time:.2f} sec")

    # ログファイルには training.log に出力されます


if __name__ == "__main__":
    main()
