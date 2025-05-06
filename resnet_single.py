from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import timm
from tqdm import tqdm
import wandb
from PIL import Image
import argparse
import os

COMPETITION_DATA_DIR = Path("data/joai-competition-2025")
NAME2LABEL = {"Mixture": 0, "NoGas": 1, "Perfume": 2, "Smoke": 3}
LABEL2NAME = {v: k for k, v in NAME2LABEL.items()}
TARGET_COL = "Gas"

@dataclass
class EnvConfig:
    data_dir: Path = COMPETITION_DATA_DIR
    image_dir: Path = data_dir / "images"
    train_path: Path = data_dir / "train.csv"
    test_path: Path = data_dir / "test.csv"
    model_save_dir: Path = Path("./")
    def __post_init__(self):
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class ExpConfig:
    seed: int = 42
    num_folds: int = 5
    batch_size: int = 64
    num_epochs: int = 100
    early_stopping_patience: int = 2
    learning_rate: float = 1e-4
    num_workers: int = 4
    img_model_name: str = "resnet50.a1h_in1k"
    add_text_model: str = "facebookai/roberta-base"

@dataclass
class Config:
    env: EnvConfig = EnvConfig()
    exp: ExpConfig = ExpConfig()

class GasModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        # Image encoder
        self.img_encoder = timm.create_model(cfg.exp.img_model_name, pretrained=True, num_classes=0, global_pool='avg')
        img_dim = self.img_encoder.num_features
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(cfg.exp.add_text_model)
        text_dim = self.text_encoder.config.hidden_size
        # Tabular input dimension computed at runtime
        self.tab_dim = None
        # MLP head
        hidden = 512
        dropout=0.2
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + text_dim, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, len(NAME2LABEL))
        )
    def forward(self, images, input_ids, attention_mask):
        img_feat = self.img_encoder.forward_features(images)
        img_feat = self.img_encoder.global_pool(img_feat).flatten(1)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.pooler_output
        combined = torch.cat([img_feat, text_feat], dim=1)
        return self.classifier(combined)

class GasDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Config, df: pd.DataFrame, transform=None, train=True):
        self.cfg, self.df, self.transform, self.train = cfg, df, transform, train
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.exp.add_text_model)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.env.image_dir / row["image_path_uuid"]).convert("RGB")
        if self.transform: img = self.transform(img)
        enc = self.tokenizer(row["Caption"] or "", padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = enc.input_ids.squeeze(0)
        attn = enc.attention_mask.squeeze(0)
        if self.train:
            label = torch.tensor(NAME2LABEL[row[TARGET_COL]], dtype=torch.long)
            return img, input_ids, attn, label
        else:
            return img, input_ids, attn

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train(); total_loss=0
    for batch in tqdm(loader, desc="Train"):
        imgs, ids, attn, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        logits = model(imgs, ids, attn)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def validate(model, loader, device):
    model.eval(); total_loss=0; preds=[]; gts=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Valid"):
            imgs, ids, attn, labels = [b.to(device) for b in batch]
            logits = model(imgs, ids, attn)
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            preds.extend(torch.argmax(logits,1).cpu().numpy())
            gts.extend(labels.cpu().numpy())
    acc = (np.array(preds)==np.array(gts)).mean()
    return total_loss/len(loader), acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    args = parser.parse_args()
    cfg=Config()
    # Load data
    train_df = pd.read_csv(args.train_csv or cfg.env.train_path)
    test_df  = pd.read_csv(args.test_csv  or cfg.env.test_path)
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # WandB
    wandb.init(project="joai-competition-2025-verification", config=vars(cfg.exp))
    # Prepare folds
    skf = StratifiedKFold(cfg.exp.num_folds, shuffle=True, random_state=cfg.exp.seed)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df[TARGET_COL])):
        tr_df, val_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        # DataLoader
        train_loader = DataLoader(GasDataset(cfg, tr_df, transform=None, train=True),
                                  batch_size=cfg.exp.batch_size, shuffle=True, num_workers=cfg.exp.num_workers)
        val_loader   = DataLoader(GasDataset(cfg, val_df, transform=None, train=True),
                                  batch_size=cfg.exp.batch_size, shuffle=False, num_workers=cfg.exp.num_workers)
        # Model
        model = GasModel(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.exp.learning_rate)
        total_steps = len(train_loader)*cfg.exp.num_epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)
        # Train/Evaluate
        best_loss=float('inf')
        for epoch in range(cfg.exp.num_epochs):
            train_loss = train_one_epoch(model, train_loader, opt, sched, device)
            val_loss, val_acc = validate(model, val_loader, device)
            wandb.log({f"fold{fold}/train_loss":train_loss, f"fold{fold}/val_loss":val_loss, f"fold{fold}/val_acc":val_acc})
            if val_loss<best_loss:
                best_loss=val_loss
                torch.save(model.state_dict(), cfg.env.model_save_dir/f"best_fold{fold}.pt")
    # Inference on test
    test_loader = DataLoader(GasDataset(cfg, test_df, transform=None, train=False),
                              batch_size=cfg.exp.batch_size, shuffle=False, num_workers=cfg.exp.num_workers)
    # Ensemble or last fold
    model = GasModel(cfg).to(device)
    model.load_state_dict(torch.load(cfg.env.model_save_dir/f"best_fold{cfg.exp.num_folds-1}.pt"))
    model.eval()
    preds=[]
    with torch.no_grad():
        for imgs, ids, attn in tqdm(test_loader, desc="Test"):
            imgs, ids, attn = imgs.to(device), ids.to(device), attn.to(device)
            logits = model(imgs, ids, attn)
            preds.extend(torch.argmax(logits,1).cpu().numpy())
    sub = pd.read_csv(COMPETITION_DATA_DIR/"sample_submission.csv")
    sub[TARGET_COL] = [LABEL2NAME[i] for i in preds]
    sub.to_csv(cfg.env.model_save_dir/"submission_single.csv", index=False)
    print("Saved single-GPU submission to submission_single.csv")

if __name__=="__main__":
    main()