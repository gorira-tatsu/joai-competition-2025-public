# table_xgb.py
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import wandb
from sklearn.metrics import accuracy_score, f1_score

from wandb import Api
from wandb.integration.xgboost import WandbCallback

from dataclasses import dataclass
from pathlib import Path

@dataclass
class EnvConfig:
    data_dir: Path = Path("data/joai-competition-2025")
    image_dir: Path = data_dir / "images"
    train_path: Path = data_dir / "train.csv"
    test_path: Path = data_dir / "test.csv"
    model_save_dir: Path = Path("./")

    def __post_init__(self):
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class ExpConfig:
    objective: str = "multi:softprob"
    num_class: int = 4
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    eval_metric: str = "mlogloss"
    n_splits: int = 5
    early_stopping_rounds: int = 20
    # regularization to reduce overfitting
    reg_alpha: float = 0.0         # L1 regularization term on weights
    reg_lambda: float = 1.0        # L2 regularization term on weights
    gamma: float = 0.0             # minimum loss reduction for split
    num_boost_round: int = 100     # number of boosting rounds

@dataclass
class Config:
    env: EnvConfig = EnvConfig()
    exp: ExpConfig = ExpConfig()


# ラベルマッピング
NAME2LABEL = {"Mixture": 0, "NoGas": 1, "Perfume": 2, "Smoke": 3}

def main(args):
    config = Config()
    wandb.init(
        project="joai-competition-2025-verification",
        job_type="train",
        config={**config.exp.__dict__, "model": "XGBClassifier"}
    )
    # Determine train file path
    if args and args.train_csv:
        train_path = args.train_csv
        print(f"Using custom train input: {train_path}")
        # Log custom train CSV as an artifact
        train_artifact = wandb.Artifact("custom_train_csv", type="dataset")
        train_artifact.add_file(str(train_path))
        wandb.log_artifact(train_artifact)
    else:
        train_path = config.env.data_dir / "train.csv"
    # Determine test file path
    if args and args.test_csv:
        test_path = args.test_csv
        print(f"Using custom test input: {test_path}")
        # Log custom test CSV as an artifact
        test_artifact = wandb.Artifact("custom_test_csv", type="dataset")
        test_artifact.add_file(str(test_path))
        wandb.log_artifact(test_artifact)
    else:
        test_path = config.env.data_dir / "test.csv"

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    # Drop unused columns
    train = train.drop(columns=["id", "Caption", "image_path_uuid"])
    test = test.drop(columns=["id", "Caption", "image_path_uuid"])

    features = [c for c in train.columns if c not in ["id", "image_path_uuid", "Caption", "Gas",]]
    X = train[features].values
    y = train["Gas"].map(NAME2LABEL).values
    X_test = test[features].values

    oof_probs = np.zeros((len(train), 4), dtype=float)
    test_probs = np.zeros((len(test), 4), dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # Prepare DMatrices for XGBoost training
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
        dval   = xgb.DMatrix(X_va, label=y_va, feature_names=features)
        params = {
            "objective": config.exp.objective,
            "num_class": config.exp.num_class,
            "learning_rate": config.exp.learning_rate,
            "max_depth": config.exp.max_depth,
            "subsample": config.exp.subsample,
            "colsample_bytree": config.exp.colsample_bytree,
            "seed": config.exp.random_state,
            "eval_metric": config.exp.eval_metric,
            "alpha":    config.exp.reg_alpha,
            "lambda":   config.exp.reg_lambda,
            "gamma":    config.exp.gamma,
        }
        # Train with early stopping
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=config.exp.num_boost_round,
            evals=[(dval, "validation")],
            early_stopping_rounds=config.exp.early_stopping_rounds,
            callbacks=[WandbCallback(log_model=True)],
            verbose_eval=False
        )
        # Log validation loss
        final_val_loss = bst.best_score
        # Compute validation predictions and additional metrics
        val_preds = bst.predict(xgb.DMatrix(X_va, feature_names=features))
        val_labels = y_va
        val_pred_labels = np.argmax(val_preds, axis=1)
        val_f1 = f1_score(val_labels, val_pred_labels, average='weighted')
        val_acc = accuracy_score(val_labels, val_pred_labels)
        wandb.log({
            "val_mlogloss": final_val_loss,
            "val_f1": val_f1,
            "val_accuracy": val_acc
        })
        # Log feature importance table and chart
        importance_dict = bst.get_score(importance_type='weight')
        importance_df = pd.DataFrame(list(importance_dict.items()), columns=['feature', 'importance'])
        # Preserve full importance list before selecting top20
        full_importance_df = importance_df.copy()
        # Select the top 20 features by importance, then sort ascending for clearer bar chart
        top20 = importance_df.nlargest(20, 'importance')
        importance_df = top20.sort_values(by='importance', ascending=True)
        imp_table = wandb.Table(dataframe=importance_df)
        wandb.log({
            "feature_importance_table": imp_table,
            "feature_importance_chart": wandb.plot.bar(
                imp_table,
                "feature",
                "importance",
                title="Top 20 Feature Importance"
            )
        })
        # Print full feature importances and save to CSV
        print(f"[Fold {fold}] Full feature importances (weight):")
        print(full_importance_df.to_string())
        full_imp_path = config.env.model_save_dir / f"feature_importance_full_fold_{fold}.csv"
        full_importance_df.to_csv(full_imp_path, index=False)
        wandb.save(str(full_imp_path))
        # Save XGBoost model weights if not running under a sweep
        if args is not None:
            model_path = config.env.model_save_dir / f"xgb_model_fold_{fold}.model"
            bst.save_model(str(model_path))
            wandb.save(str(model_path))
        # Predict
        oof_probs[va_idx] = bst.predict(xgb.DMatrix(X_va, feature_names=features))
        test_probs += bst.predict(xgb.DMatrix(X_test, feature_names=features)) / skf.n_splits
        print(f"[Fold {fold}] done")

    # 保存
    np.save(config.env.model_save_dir / "oof_xgb.npy", oof_probs)
    np.save(config.env.model_save_dir / "test_xgb.npy", test_probs)
    wandb.save(str(config.env.model_save_dir / "oof_xgb.npy"))
    wandb.save(str(config.env.model_save_dir / "test_xgb.npy"))
    print("Saved oof_xgb.npy and test_xgb.npy")

def train():
    # Initialize a W&B run under the sweep context
    wandb.init()
    # Override experiment config with sweep parameters
    config = Config()
    for param, value in wandb.config.items():
        if hasattr(config.exp, param):
            setattr(config.exp, param, value)
    # Run the main training logic using updated config
    main(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=False,
                        default=Path("data/joai-competition-2025"),
                        help="Path to data folder containing train.csv/test.csv (default: data/joai-competition-2025)")
    parser.add_argument("--train_csv", type=Path, help="Path to custom train CSV", default=None)
    parser.add_argument("--test_csv", type=Path, help="Path to custom test CSV", default=None)
    args = parser.parse_args()
    main(args)