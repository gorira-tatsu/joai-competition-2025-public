from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import timm
from tqdm import tqdm
import wandb
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import argparse
import os

COMPETITION_DATA_DIR = Path("data/joai-competition-2025")
train_df = pd.read_csv(COMPETITION_DATA_DIR / 'train.csv')
test_df = pd.read_csv(COMPETITION_DATA_DIR / 'test.csv')
sample_submission_df = pd.read_csv(COMPETITION_DATA_DIR / 'sample_submission.csv')
NAME2LABEL = {"Mixture": 0, "NoGas": 1, "Perfume": 2, "Smoke": 3}
NUM_CLASSES = len(NAME2LABEL)
TARGET_COL = "Gas"


def prepare_folds(train_df: pd.DataFrame, cfg: 'Config') -> pd.DataFrame:
    fold_array = np.zeros(len(train_df), dtype=np.int32)
    skf = StratifiedKFold(
        n_splits=cfg.exp.num_folds,
        shuffle=True,
        random_state=cfg.exp.seed
    )
    for fold, (_, valid_idx) in enumerate(skf.split(train_df, train_df[TARGET_COL])):
        fold_array[valid_idx] = fold
    train_df['fold'] = fold_array
    return train_df


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
    seed: int = 42
    num_folds: int = 5
    batch_size: int = 64
    num_epochs: int = 100
    early_stopping_patience: int = 2
    learning_rate: float = 1e-4
    num_workers: int = 4
    img_model_name: str = "resnet50.a1h_in1k"
    img_pretrained: bool = True
    text_model_name: str = "facebookai/roberta-base"
    text_pretrained: bool = True
    tabular_cols: list = field(default_factory=lambda: [
        col for col in train_df.columns
        if col not in ["Caption", "image_path_uuid", TARGET_COL]
    ])

@dataclass
class Config:
    env: EnvConfig = EnvConfig()
    exp: ExpConfig = ExpConfig()

class GasModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.img_encoder = timm.create_model(
            cfg.exp.img_model_name,
            pretrained=cfg.exp.img_pretrained,
            num_classes=0,  # remove classification head
            global_pool='avg'
        )
        img_feat_dim = self.img_encoder.num_features
        self.text_encoder = AutoModel.from_pretrained(
            cfg.exp.text_model_name,
            add_pooling_layer=True
        )
        text_feat_dim = self.text_encoder.config.hidden_size

        # MLP head hyperparams 
        # Default values for hyperparameters if we're not getting them from wandb
        hidden_dim = 512
        dropout_rate = 0.2
        
        # Try to get values from wandb if available, or use defaults if not
        try:
            if hasattr(wandb, 'config'):
                hidden_dim = getattr(wandb.config, "mlp_hidden_dim", hidden_dim)
                dropout_rate = getattr(wandb.config, "mlp_dropout", dropout_rate)
        except Exception:
            # Use default values if anything goes wrong with wandb
            pass

        # Tabular MLP branch with BatchNorm and dropout for regularization
        tab_hidden = 128
        self.tabular_mlp = nn.Sequential(
            nn.Linear(len(cfg.exp.tabular_cols), tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(tab_hidden, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Linear(img_feat_dim + text_feat_dim + tab_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, NUM_CLASSES)
        )

    def forward(self, images, input_ids, attention_mask, tabular_feats):
        features = self.img_encoder.forward_features(images)
        img_feats = self.img_encoder.global_pool(features)
        img_feats = img_feats.flatten(1)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_outputs.pooler_output
        tab_feats = self.tabular_mlp(tabular_feats)
        combined = torch.cat([img_feats, text_feats, tab_feats], dim=1)
        return self.classifier(combined)

class TrainGasDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Config, df: pd.DataFrame, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.cfg.env.image_dir / self.df["image_path_uuid"][index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_str = self.df["Gas"][index]
        label = torch.tensor(NAME2LABEL[label_str], dtype=torch.long)
        encoded = tokenizer(
            self.df["Caption"][index],
            padding='max_length', truncation=True, max_length=128, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        # extract tabular features
        tabular_values = torch.tensor(
            [self.df[col].iloc[index] for col in self.cfg.exp.tabular_cols],
            dtype=torch.float
        )
        return image, input_ids, attention_mask, tabular_values, label

def train_one_epoch(model, train_loader, optimizer_img, optimizer_text, criterion, device, scheduler_text, fold, local_rank):
    model.train()
    total_train_loss = 0.0
    batch_count = 0
    for images, input_ids, attention_mask, tabular_feats, labels in tqdm(train_loader, desc="Training", disable=local_rank != 0):
        images, input_ids, attention_mask, tabular_feats, labels = images.to(device), input_ids.to(device), attention_mask.to(device), tabular_feats.to(device), labels.to(device)
        optimizer_img.zero_grad()
        optimizer_text.zero_grad()
        outputs = model(images, input_ids, attention_mask, tabular_feats)
        loss = criterion(outputs, labels)
        if local_rank == 0:
            try:
                wandb.log({f"fold{fold}/train_loss": loss.item()})
            except Exception:
                print(f"Fold {fold}, Batch {batch_count}, Train Loss: {loss.item():.4f}")
        total_train_loss += loss.item()
        batch_count += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_img.step()
        optimizer_text.step()
        scheduler_text.step()
    avg_train_loss = total_train_loss / batch_count
    return avg_train_loss

def validate(model, valid_loader, criterion, device, fold, local_rank):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, input_ids, attention_mask, tabular_feats, labels in tqdm(valid_loader, desc="Validation", disable=local_rank != 0):
            images, input_ids, attention_mask, tabular_feats, labels = images.to(device), input_ids.to(device), attention_mask.to(device), tabular_feats.to(device), labels.to(device)
            outputs = model(images, input_ids, attention_mask, tabular_feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # gather predictions and labels from all processes
    preds_tensor = torch.tensor(all_preds, device=device)
    labels_tensor = torch.tensor(all_labels, device=device)
    gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_preds, preds_tensor)
    dist.all_gather(gathered_labels, labels_tensor)
    if local_rank == 0:
        all_preds = torch.cat(gathered_preds).cpu().numpy().tolist()
        all_labels = torch.cat(gathered_labels).cpu().numpy().tolist()
    dist.barrier() 

    avg_loss = total_loss / len(valid_loader)
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted') if local_rank == 0 else 0.0
    
    if local_rank == 0:
        try:
            wandb.log({
                f"fold{fold}/valid_loss": avg_loss,
                f"fold{fold}/valid_accuracy": accuracy,
                f"fold{fold}/valid_f1": f1
            })
        except Exception:
            print(f"Fold {fold}, Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return avg_loss, accuracy, f1

class TestGasDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Config, df: pd.DataFrame, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.cfg.env.image_dir / self.df["image_path_uuid"][index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        encoded = tokenizer(
            self.df["Caption"][index],
            padding='max_length', truncation=True, max_length=128, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        tabular_values = torch.tensor(
            [self.df[col].iloc[index] for col in self.cfg.exp.tabular_cols],
            dtype=torch.float
        )
        return image, input_ids, attention_mask, tabular_values

def run_inference(model, test_loader, device, local_rank):
    model.eval()
    probs = []
    with torch.no_grad():
        for images, input_ids, attention_mask, tabular_feats in tqdm(test_loader, disable=local_rank != 0):
            images, input_ids, attention_mask, tabular_feats = images.to(device), input_ids.to(device), attention_mask.to(device), tabular_feats.to(device)
            outputs = model(images, input_ids, attention_mask, tabular_feats)
            probs.append(torch.softmax(outputs, dim=1).cpu())
    probs = torch.cat(probs, dim=0).numpy()
    return probs

def setup_ddp(local_rank):
    """
    Set up distributed training environment
    """
    # Initialize the process group
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return torch.device(f"cuda:{local_rank}")

def cleanup_ddp():
    """
    Clean up distributed training environment
    """
    dist.destroy_process_group()

def main():
    import torch.optim as optim
    from torchvision import transforms
    import numpy as np
    
    import argparse
    # Parse command-line overrides for data paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default=None, help="Path to train CSV file")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to test CSV file")
    args = parser.parse_args()

    # Determine local rank from environment
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    # Override default dataframes if provided
    global train_df, test_df
    if args.train_csv is not None:
        train_df = pd.read_csv(args.train_csv)
    if args.test_csv is not None:
        test_df = pd.read_csv(args.test_csv)
        
    # Set up distributed training
    device = setup_ddp(local_rank)
    
    # Initialize wandb only on the main process - no special conditions
    if local_rank == 0:
        try:
            wandb.init(project="joai-competition-2025-verification")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")
    
    # Make sure all other processes wait for rank 0 to initialize wandb
    dist.barrier()
    
    cfg = Config()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.exp.text_model_name)

    # Set a unique seed per process by adding local_rank to the base seed
    base_seed = cfg.exp.seed
    cfg.exp.seed = base_seed + local_rank
    torch.manual_seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and prepare dataframes
    full_train_df = train_df.copy()
    full_train_df = prepare_folds(full_train_df, cfg)
    all_val_losses = []
    from torch.utils.data import DataLoader

    # Initialize wandb config on the main process
    if local_rank == 0:
        try:
            wandb.config.update({
                "img_model_name": cfg.exp.img_model_name,
                "text_model_name": cfg.exp.text_model_name,
                "learning_rate": cfg.exp.learning_rate,
                "batch_size": cfg.exp.batch_size,
                "num_epochs": cfg.exp.num_epochs,
                "num_folds": cfg.exp.num_folds,
                "early_stopping_patience": cfg.exp.early_stopping_patience,
                "seed": base_seed,
                "world_size": dist.get_world_size(),
            })
        except Exception:
            pass
    
    # Wait for rank 0 to initialize wandb before proceeding
    dist.barrier()
    
    # Create a dummy wandb.config for non-rank-0 processes if needed
    if local_rank != 0:
        try:
            class DummyConfig:
                def __getattr__(self, name):
                    return None
            # Replace wandb.config with a dummy for non-main processes
            if not hasattr(wandb, 'config') or wandb.config is None:
                wandb.config = DummyConfig()
        except Exception:
            pass

    resnet_lr = cfg.exp.learning_rate * 10
    resnet_momentum = 0.9
    resnet_weight_decay = 1e-4
    text_lr = cfg.exp.learning_rate
    text_weight_decay = 1e-5
    
    # wandbのconfigから値を取得（可能な場合）
    try:
        resnet_lr = getattr(wandb.config, "resnet_lr", resnet_lr)
        resnet_momentum = getattr(wandb.config, "resnet_momentum", resnet_momentum)
        resnet_weight_decay = getattr(wandb.config, "resnet_weight_decay", resnet_weight_decay)
        text_lr = getattr(wandb.config, "text_lr", text_lr)
        text_weight_decay = getattr(wandb.config, "text_weight_decay", text_weight_decay)
    except Exception:
        # wandbからの値取得に失敗した場合はデフォルト値を使用
        print(f"[Rank {local_rank}] Using default hyperparameters")

    # track overall best model across folds
    best_overall_loss = float('inf')
    best_overall_state = None
    best_overall_fold = None

    for fold in range(cfg.exp.num_folds):
        if local_rank == 0:
            print(f"Starting fold {fold}")
            
        train_fold_df = full_train_df[full_train_df['fold'] != fold].reset_index(drop=True)
        valid_fold_df = full_train_df[full_train_df['fold'] == fold].reset_index(drop=True)
        train_dataset = TrainGasDataset(cfg, train_fold_df, transform=train_transform)
        valid_dataset = TrainGasDataset(cfg, valid_fold_df, transform=test_transform)
        
        # Use DistributedSampler for the training dataset
        train_sampler = DistributedSampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.exp.batch_size,
            shuffle=False,  # Disabled when using DistributedSampler
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
            sampler=train_sampler
        )
        
        # Validation loader uses DistributedSampler for distributed evaluation
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.exp.batch_size,
            sampler=valid_sampler,
            num_workers=cfg.exp.num_workers,
            pin_memory=True
        )
        
        model = GasModel(cfg).to(device)
        # Wrap the model with DistributedDataParallel
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        # set up separate optimizers
        optimizer_img = optim.SGD(
            model.module.img_encoder.parameters(),
            lr=resnet_lr,
            momentum=resnet_momentum,
            weight_decay=resnet_weight_decay
        )
        optimizer_text = AdamW(
            [
                {"params": model.module.text_encoder.parameters(), "lr": text_lr, "weight_decay": text_weight_decay},
                {"params": model.module.tabular_mlp.parameters(), "lr": text_lr, "weight_decay": text_weight_decay},
                {"params": model.module.classifier.parameters(), "lr": text_lr, "weight_decay": text_weight_decay}
            ]
        )
        
        # set up learning rate scheduler for text optimizer
        total_steps = len(train_loader) * cfg.exp.num_epochs
        scheduler_text = get_linear_schedule_with_warmup(
            optimizer_text,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        if local_rank == 0:
            try:
                wandb.watch(model, log="all")
            except Exception:
                pass
                
        best_valid_loss = float('inf')
        patience_counter = 0
        best_train_loss = None
        best_valid_accuracy = None
        best_valid_f1 = None
        best_model_state = None
        
        for epoch in range(cfg.exp.num_epochs):
            # Set the epoch for the train sampler
            train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, optimizer_img, optimizer_text, criterion, device, scheduler_text, fold, local_rank)
            val_loss, val_acc, val_f1 = validate(model, valid_loader, criterion, device, fold, local_rank)

            # All processes need to reach consensus on early stopping
            val_loss_tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_train_loss = train_loss
                best_valid_accuracy = val_acc
                best_valid_f1 = val_f1
                # Save model state in the main process only
                if local_rank == 0:
                    best_model_state = model.module.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.exp.early_stopping_patience:
                if local_rank == 0:
                    print(f"Early stopping at fold {fold}, epoch {epoch}")
                break

            # Synchronize all processes after each epoch
            dist.barrier()
            
        # Update all_val_losses in the main process only
        if local_rank == 0:
            all_val_losses.append(best_valid_loss)
            
            # update overall best model if this fold is the best so far
            if best_valid_loss < best_overall_loss:
                best_overall_loss = best_valid_loss
                best_overall_state = best_model_state
                best_overall_fold = fold
                
            # record per-fold summary metrics
            try:
                wandb.run.summary[f"fold{fold}/train_loss"] = best_train_loss
                wandb.run.summary[f"fold{fold}/valid_loss"] = best_valid_loss
                wandb.run.summary[f"fold{fold}/valid_accuracy"] = best_valid_accuracy
                wandb.run.summary[f"fold{fold}/valid_f1"] = best_valid_f1
            except Exception:
                print(f"Fold {fold} Results:")
                print(f"  Best Train Loss: {best_train_loss:.4f}")
                print(f"  Best Valid Loss: {best_valid_loss:.4f}")
                print(f"  Best Valid Accuracy: {best_valid_accuracy:.4f}")
                print(f"  Best Valid F1: {best_valid_f1:.4f}")

            # save this fold's best model for ensembling
            fold_ckpt = cfg.env.model_save_dir / f"best_model_fold{fold}.pt"
            torch.save(best_model_state, fold_ckpt)
            try:
                wandb.save(str(fold_ckpt))
            except Exception:
                pass

            # Generate and save per-fold submission
            test_dataset = TestGasDataset(cfg, test_df, transform=test_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.exp.batch_size,
                shuffle=False,
                num_workers=cfg.exp.num_workers,
                pin_memory=True
            )
            # Load best fold model and run inference
            model.module.load_state_dict(torch.load(fold_ckpt))
            fold_probs = run_inference(model, test_loader, device, local_rank)
            fold_preds = np.argmax(fold_probs, axis=1)
            LABEL2NAME = {v: k for k, v in NAME2LABEL.items()}
            fold_submission = sample_submission_df.copy()
            fold_submission[TARGET_COL] = [LABEL2NAME[i] for i in fold_preds]
            submission_path_fold = cfg.env.model_save_dir / f"submission_fold{fold}.csv"
            fold_submission.to_csv(submission_path_fold, index=False)
            try:
                wandb.save(str(submission_path_fold))
            except Exception:
                pass
        
        # Synchronize all processes after each fold
        dist.barrier()

    # Do inference only on the main process
    if local_rank == 0:
        # === Ensemble predictions from each fold ===
        from torch.utils.data import DataLoader
        avg_val_loss = sum(all_val_losses) / len(all_val_losses)
        print(f"Average CV validation loss: {avg_val_loss}")
        try:
            wandb.log({"avg_cv_valid_loss": avg_val_loss})
        except Exception:
            pass
        test_dataset = TestGasDataset(cfg, test_df, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.exp.batch_size,
            shuffle=False,
            num_workers=cfg.exp.num_workers,
            pin_memory=True
        )
        # collect fold-level probabilities
        probs_list = []
        for fold in range(cfg.exp.num_folds):
            fold_ckpt = cfg.env.model_save_dir / f"best_model_fold{fold}.pt"
            print(f"Loading model for fold {fold} from {fold_ckpt}")
            model.module.load_state_dict(torch.load(fold_ckpt))
            probs = run_inference(model, test_loader, device, local_rank)
            probs_list.append(probs)
        # average probabilities
        avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
        # save ensemble probabilities
        ensemble_probs_path = cfg.env.model_save_dir / "probs_ensemble.npy"
        np.save(ensemble_probs_path, avg_probs)
        try:
            wandb.save(str(ensemble_probs_path))
        except Exception:
            pass
        # build ensembled submission
        submission = sample_submission_df.copy()
        LABEL2NAME = {v: k for k, v in NAME2LABEL.items()}
        preds = np.argmax(avg_probs, axis=1)
        submission[TARGET_COL] = [LABEL2NAME[i] for i in preds]
        submission_path = cfg.env.model_save_dir / "submission_ensemble.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Ensembled submission file saved to: {submission_path.resolve()}")
        try:
            wandb.save(str(submission_path))
            wandb.finish()
        except Exception:
            pass
    
    # Clean up the distributed environment
    cleanup_ddp()

if __name__ == "__main__":
    main()