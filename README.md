<!-- https://wandb.ai/gorira/joai-competition-2025/runs/60k9k87l/overview と https://wandb.ai/gorira/joai-competition-2025/runs/rgbdc2gn/overview を利用した。もし必要であれば運営に共有 -->

## Setup and Usage

1. 環境構築: `uv` で環境構築を行い、`uv sync` を実行すること

2. 仮想環境アクティベート: `source .venv/bin/activate`

3. W&Bログイン: `wandb login`

4. データ配置: `data` ディレクトリ内に `joai-competition-2025` フォルダを配置すること

5. 特徴量生成: 以下のコマンドを実行して特徴量ファイルを生成すること
   ```bash
   python FeatureEngneering.py
   ```
   - 出力: `train_nn_features.csv` と `test_nn_features.csv` がプロジェクトルートに作成されます

6. モデル訓練実行:
   - 分散トレーニング (複数GPU):
     ```bash
     torchrun resnet.py --train_csv ./train_nn_features.csv --test_csv ./test_nn_features.csv
     ```
   - 単一GPUトレーニング (マルチGPUが難しい場合):
     ```bash
     python resnet_single.py --train_csv ./train_nn_features.csv --test_csv ./test_nn_features.csv
     ```
     - 注意: このスクリプトはマルチGPU版から専用に書き直しているため、完全な動作同一性は保証しません

7. XGBoostモデル実行:
   ```bash
   python table_xgb.py --train_csv ./train_nn_features.csv --test_csv ./test_nn_features.csv
   ```

**補足:**  
- PyTorchや関連パッケージで問題が発生した場合は、`uv pip install <パッケージ名> --upgrade` を使って再インストールしてください。