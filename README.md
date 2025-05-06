<!-- https://wandb.ai/gorira/joai-competition-2025/runs/60k9k87l/overview と https://wandb.ai/gorira/joai-competition-2025/runs/rgbdc2gn/overview を利用した。もし必要であれば運営に共有 -->

# JOAI2025 gorira-tatsu / 秋山達彦 の再現実装の実行手順

このプロジェクトはJOAI参加者である gorira-tatsu / 秋山達彦 のGitの履歴、Kaggleの提出記録、W&Bのログ、JOAI競技時のローカルのメモをもとに、`Mon Apr 28 2025 22:54:44 GMT+0900 (Japan Standard Time)`に提出したBest Subを再現実装したものである。

ここに記載されたプログラムは、不要なコメント、デバッグ用のコード等を削除し、忠実に再現されたものである。

プロジェクトツリー(後述の通り、dataディレクトリに`joai-competition-2025`という名前でデータを保存してください。)
```bash
joai-competition-2025-public on  main [!] is 📦 v0.1.0 via 🐍 v3.10.10 on ☁️  tatsuhiko.shigoto@gmail.com
❯ tree -L 3
.
├── FeatureEngneering.py
├── README.md
├── data
│   ├── README.md
│   └── joai-competition-2025
│       ├── images
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── ensamble
│   └── README.md
├── ensamble.py
├── pyproject.toml
├── resnet.py
├── resnet_single.py
├── table_xgb.py
└── uv.lock

5 directories, 13 files
```

以下、再現方法である。

## Setup and Usage

1. 環境構築: `uv` で環境構築を行い、`uv sync` を実行する

2. 仮想環境をアクティベートする: `source .venv/bin/activate`

3. W&Bログイン: `wandb login`

4. データ配置: `data` ディレクトリ内に `joai-competition-2025` フォルダを配置すること

5. 特徴量生成: 以下のコマンドを実行して特徴量ファイルを生成すること
   ```bash
   python FeatureEngneering.py
   ```
   - 出力: `train_nn_features.csv` と `test_nn_features.csv` がプロジェクトルートディレクトリに作成されます

6. Resnetモデル訓練実行:
   - 分散トレーニング (複数GPU):
     ```bash
     torchrun resnet.py --train_csv ./train_nn_features.csv --test_csv ./test_nn_features.csv
     ```
   - 単一GPUトレーニング (マルチGPUが難しい場合):
     ```bash
     python resnet_single.py --train_csv ./train_nn_features.csv --test_csv ./test_nn_features.csv
     ```
     - 注意: このスクリプトはマルチGPU版から専用に書き直しているため、完全な動作同一性は保証しません。しかしながら、事後検証でKaggle提出版csvとの差はほとんどない(99%の同一性)ことが確認できています。

7. XGBoostモデル訓練実行:
   ```bash
   python table_xgb.py --train_csv ./train_nn_features.csv --test_csv ./test_nn_features.csv
   ```

8. アンサンブルの実行:  
   Resnetで生成された`probs_ensemble.npy`とXGBoostで生成された`test_xgb.npy`を`ensamble`ディレクトリにコピーする

   ```bash
   cp probs_ensemble.npy ensamble/
   cp test_xgb.npy ensamble/
   ```

   アンサンブルを実行する

   ```bash
   python ensamble.py
   ```

   アンサンブルによって`submission_ensemble.csv`が生成されることを確認する

   ```bash
   $ ls | grep submission_ensemble.csv
   # submission_ensemble.csv
   ```


**補足:**  
- PyTorchや関連パッケージで問題が発生した場合は、`uv pip install <パッケージ名> --upgrade` を使って再インストールしてください。