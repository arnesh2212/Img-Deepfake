import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import argparse
import json
from datetime import datetime
import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as T
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torchmetrics
    USE_TORCHMETRICS = True
except ImportError:
    print("WARNING: torchmetrics not found. Pixel-level AUC/F1 will not be calculated.")
    USE_TORCHMETRICS = False

from src.utils import SIDADataset
from src.models import (FreqDINO, FreqDINO_ablation1, FreqDINO_ablation2, 
                        FreqDINO_ablation3, FreqDINO_ablation4, FreqDINO_ablation5, FreqDINO_ablation0)
from src.loss import CombinedSegmentationLoss, CrossModalContrastiveLoss
import random
from tqdm.auto import tqdm
from datasets import load_from_disk
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


# Argument parser
parser = argparse.ArgumentParser(description="Train and Test FreqDINO Ablations")
parser.add_argument('--model', type=str, default='baseline', 
                    choices=['baseline', 'ablation1', 'ablation2', 'ablation3', 'ablation4', 'ablation5', 'ablation0'],
                    help='Model variant to train')
parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
parser.add_argument('--gpu', type=str, default='3', help='GPU device ID')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Model mapping
MODEL_MAP = {
    'baseline': FreqDINO,
    'ablation0': FreqDINO_ablation0,
    'ablation1': FreqDINO_ablation1,
    'ablation2': FreqDINO_ablation2,
    'ablation3': FreqDINO_ablation3,
    'ablation4': FreqDINO_ablation4,
    'ablation5': FreqDINO_ablation5,
}

# Configuration
LOCAL_DS_DIR = Path("/home/arush/deepfake/sida_net/datasets/SID_Set")
EMBEDDINGS_FOLDER_train = "/home/arush/deepfake/sida_net/embeddings/train"
EMBEDDINGS_FOLDER_val = "/home/arush/deepfake/sida_net/embeddings/validation"
EMBEDDINGS_FOLDER_test = "/home/arush/deepfake/sida_net/test_download/test/embeddings/test"

# Determine run name
if args.run_name is None:
    args.run_name = f"freq_dino_{args.model}"

# Experiment directory
EXPERIMENT_DIR = Path(f"./experiments/{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
LOGS_DIR = EXPERIMENT_DIR / "logs"
TEST_RESULTS_DIR = EXPERIMENT_DIR / "test_results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"

for d in [CHECKPOINT_DIR, LOGS_DIR, TEST_RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    "project": "deepfake-freq-dinov3-ablations",
    "entity": None,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "val_batch_size": 32,
    "lr": 1e-4,
    "seg_lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 8,
    "log_interval": 50,
    "img_size": 224,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seg_loss_weight": 1.0,
    "cls_loss_weight": 1.0,
    "contrast_loss_weight": 0.3,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "checkpoint_dir": str(CHECKPOINT_DIR),
    "logs_dir": str(LOGS_DIR),
    "test_results_dir": str(TEST_RESULTS_DIR),
    "plots_dir": str(PLOTS_DIR),
    "project_tags": ["freq", "dino", "clip", "segmentation", "ablation"],
    "num_classes": 3,
    "run_name": args.run_name,
    "model_type": args.model,
}

# Save config
with open(LOGS_DIR / "config.json", "w") as f:
    json.dump(CFG, f, indent=4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])


def _to_mask_tensor(m, img_size):
    if m is None:
        return None

    def _resize_nn(x):
        if x.shape[-2:] != (img_size, img_size):
            x = TF.resize(x, [img_size, img_size], interpolation=InterpolationMode.NEAREST)
        return x

    if isinstance(m, Image.Image):
        t = TF.to_tensor(m)
        if t.size(0) > 1:
            t = t.mean(dim=0, keepdim=True)
        else:
            t = t
        t = _resize_nn(t)
        t = (t > 0.5).float()
        return t.squeeze(0)

    if isinstance(m, np.ndarray):
        t = torch.from_numpy(m).float()
        if t.ndim == 3:
            t = t.mean(dim=2)
        if t.max() > 1.0:
            t = t / 255.0
        t = t.clamp(0,1).unsqueeze(0)
        t = _resize_nn(t)
        t = (t > 0.5).float()
        return t.squeeze(0)

    if torch.is_tensor(m):
        t = m.float()
        if t.ndim == 3:
            t = t.mean(dim=0)
        if t.max() > 1.0:
            t = t / 255.0
        t = t.clamp(0,1).unsqueeze(0)
        t = _resize_nn(t)
        t = (t > 0.5).float()
        return t.squeeze(0)

    return None


def collate_fn(batch):
    b_img = []
    b_clip = []
    b_dino_cls = []
    b_dino_reg = []
    b_dino_patch = []
    b_label = []
    masks = []

    for s in batch:
        feats = s["features"]
        b_img.append(feats["image"])
        b_clip.append(feats["clip_embed"])
        b_dino_cls.append(feats["dino_cls"])
        b_dino_reg.append(feats.get(
            "dino_reg",
            torch.zeros((4, feats["dino_cls"].shape[-1]), dtype=feats["dino_cls"].dtype)
        ))
        b_dino_patch.append(feats["dino_patch"])
        b_label.append(torch.tensor(s["label"], dtype=torch.long))
        masks.append(s.get("mask", None))

    images = torch.stack(b_img, dim=0)
    clip = torch.stack(b_clip, dim=0)
    dino_cls = torch.stack(b_dino_cls, dim=0)
    dino_reg = torch.stack(b_dino_reg, dim=0)
    dino_patch = torch.stack(b_dino_patch, dim=0)
    labels = torch.stack(b_label).long()

    has_mask = torch.tensor([1 if m is not None else 0 for m in masks], dtype=torch.uint8)

    valid_mask_indices = []
    processed_masks = []
    
    for i, m in enumerate(masks):
        mt = _to_mask_tensor(m, CFG["img_size"])
        if mt is not None and has_mask[i].item() == 1:
            valid_mask_indices.append(i)
            processed_masks.append(mt)
    
    if len(processed_masks) > 0:
        masks_t = torch.stack(processed_masks, dim=0)
    else:
        masks_t = None
    
    return {
        "image": images,
        "clip_embed": clip,
        "dino_cls": dino_cls,
        "dino_reg": dino_reg,
        "dino_patch": dino_patch,
        "label": labels,
        "has_mask": has_mask,
        "mask": masks_t,
        "valid_mask_indices": torch.tensor(valid_mask_indices) if len(valid_mask_indices) > 0 else None
    }


def iou_score(pred_mask, true_mask, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_mask) > thr).float()
    inter = (pred * true_mask).sum()
    union = pred.sum() + true_mask.sum() - inter
    if union.item() == 0:
        return torch.tensor(1.0) if inter.item() == 0 else torch.tensor(0.0)
    return (inter + eps) / (union + eps)


def batch_iou(sig_logits, gt_masks, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(sig_logits) > thr).float()
    gt   = (gt_masks > 0.5).float()

    inter = (pred * gt).flatten(1).sum(dim=1)
    union = pred.flatten(1).sum(dim=1) + gt.flatten(1).sum(dim=1) - inter
    iou   = torch.where(union > 0, (inter + eps)/(union + eps), (inter == 0).float())
    return iou


def compute_macro_f1_torch(preds, targets, num_classes):
    preds = preds.view(-1)
    targets = targets.view(-1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(sum(f1s) / len(f1s))


# Load datasets
BASE_DATA = load_from_disk(str(LOCAL_DS_DIR))

transforms_data = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.ToTensor(),
])

train_ds = SIDADataset(BASE_DATA, EMBEDDINGS_FOLDER_train, split='train', transform=transforms_data)
val_ds = SIDADataset(BASE_DATA, EMBEDDINGS_FOLDER_val, split='validation', transform=transforms_data)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CFG["val_batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], collate_fn=collate_fn, pin_memory=True)

# Initialize model
device = torch.device(CFG["device"])
ModelClass = MODEL_MAP[args.model]
model = ModelClass(num_classes=CFG["num_classes"]).to(device)

# Optimizers
cls_params = [p for n, p in model.named_parameters() if 'seg_decoder' not in n]
seg_params = [p for n, p in model.named_parameters() if 'seg_decoder' in n]

optimizer = torch.optim.AdamW([
    {'params': cls_params, 'lr': CFG["lr"]},
    {'params': seg_params, 'lr': CFG["seg_lr"]}
], weight_decay=CFG["weight_decay"])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# Losses
criterion_cls = nn.CrossEntropyLoss()
criterion_seg = CombinedSegmentationLoss(
    focal_weight=1.0,
    dice_weight=1.0,
    focal_alpha=CFG["focal_alpha"],
    focal_gamma=CFG["focal_gamma"]
)
criterion_contrast = CrossModalContrastiveLoss(temperature=0.1)

# WandB
wandb_mode = os.getenv("WANDB_MODE", "online")
wandb.init(project=CFG["project"], entity=CFG["entity"], config=CFG, 
           tags=CFG["project_tags"], mode=wandb_mode, name=CFG['run_name'])
wandb.watch(model, log="all", log_freq=100)

# Training log file
train_log_file = LOGS_DIR / "training_log.txt"

def log_message(msg):
    print(msg)
    with open(train_log_file, "a") as f:
        f.write(msg + "\n")


def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    running_contrast_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    seg_samples = 0
    
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        images = batch["image"].to(device)
        clip_embed = batch["clip_embed"].to(device)
        dino_cls = batch["dino_cls"].to(device)
        dino_reg = batch["dino_reg"].to(device)
        dino_patch = batch["dino_patch"].to(device)
        labels = batch["label"].to(device)
        has_mask = batch["has_mask"].to(device)
        mask = batch["mask"].to(device) if batch["mask"] is not None else None
        valid_mask_indices = batch["valid_mask_indices"]
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(images, clip_embed, dino_cls, dino_reg, dino_patch, 
                       has_mask=has_mask, mask=mask)
            
            logits = out["logits"]
            seg_logits = out["seg_logits"]
            
            cls_loss = criterion_cls(logits, labels) * CFG["cls_loss_weight"]
            total_loss = cls_loss
            
            seg_loss_val = 0.0
            if mask is not None and valid_mask_indices is not None and len(valid_mask_indices) > 0:
                seg_pred = seg_logits[valid_mask_indices]
                seg_gt = mask.unsqueeze(1)
                
                seg_loss = criterion_seg(seg_pred, seg_gt)
                seg_loss_val = seg_loss.item()
                total_loss = total_loss + CFG["seg_loss_weight"] * seg_loss
                seg_samples += len(valid_mask_indices)
                
            #only for ablation with contrastive loss (ablation 0 that is
            contrast_loss_val = 0.0
            if "noise_contrast" in out and "dino_contrast" in out:
                noise_feats = out["noise_contrast"]
                dino_feats = out["dino_contrast"]
                
                contrast_loss = criterion_contrast(noise_feats, dino_feats, labels)
                contrast_loss_val = contrast_loss.item()  
                total_loss = total_loss + CFG["contrast_loss_weight"] * contrast_loss
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        running_seg_loss += seg_loss_val * images.size(0)
        
        running_contrast_loss += contrast_loss_val * images.size(0)
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        if (step + 1) % CFG["log_interval"] == 0:
            log_dict = {
                "train/lr" : optimizer.param_groups[0]["lr"],
                "train/seg_lr": optimizer.param_groups[1]["lr"],
                "train/step_loss": total_loss.item(),
                "train/cls_loss": cls_loss.item(),
                "train/seg_loss": seg_loss_val,
                "epoch": epoch,
                "step": epoch * len(train_loader) + step
            }
            if contrast_loss_val > 0.0:
                log_dict["train/contrast_loss"] = contrast_loss_val
            wandb.log(log_dict)
    
    avg_loss = running_loss / total
    avg_cls = running_cls_loss / total
    avg_seg = (running_seg_loss / total) if seg_samples > 0 else 0.0
    avg_contrast = running_contrast_loss / total
    acc = correct / total
    
    preds_cat = torch.cat(all_preds, dim=0).numpy()
    labels_cat = torch.cat(all_labels, dim=0).numpy()
    
    try:
        train_f1_macro = f1_score(labels_cat, preds_cat, average="macro")
        train_f1_micro = f1_score(labels_cat, preds_cat, average="micro")
    except Exception:
        train_f1_macro = compute_macro_f1_torch(torch.tensor(preds_cat), 
                                         torch.tensor(labels_cat), CFG["num_classes"])
        train_f1_micro = 0.0
    
    log_dict = {
        "train/epoch_loss": avg_loss,
        "train/cls_loss_epoch": avg_cls,
        "train/seg_loss_epoch": avg_seg,
        "train/acc": acc,
        "train/f1_macro": train_f1_macro,
        "train/f1_micro": train_f1_micro,
        "train/seg_samples": seg_samples,
        "epoch": epoch
    }
    if avg_contrast > 0.0:
        log_dict["train/contrast_loss_epoch"] = avg_contrast
    
    wandb.log(log_dict)
    
    if avg_contrast > 0.0:
        msg = f"Epoch {epoch} Train - Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1_macro: {train_f1_macro:.4f}, F1_micro: {train_f1_micro:.4f}, Contrast_Loss: {avg_contrast:.4f}"
    else:
        msg = f"Epoch {epoch} Train - Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1_macro: {train_f1_macro:.4f}, F1_micro: {train_f1_micro:.4f}"
    log_message(msg)
    
    return avg_loss, acc, train_f1_macro


@torch.no_grad()
def validate(epoch, num_visuals=4):
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    iou_sum = 0.0
    iou_count = 0
    visuals = []
    seg_samples = 0
    
    for step, batch in enumerate(tqdm(val_loader, desc=f"Val Epoch {epoch}")):
        images = batch["image"].to(device)
        clip_embed = batch["clip_embed"].to(device)
        dino_cls = batch["dino_cls"].to(device)
        dino_reg = batch["dino_reg"].to(device)
        dino_patch = batch["dino_patch"].to(device)
        labels = batch["label"].to(device)
        has_mask = batch["has_mask"].to(device)
        mask = batch["mask"].to(device) if batch["mask"] is not None else None
        valid_mask_indices = batch["valid_mask_indices"]
        
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(images, clip_embed, dino_cls, dino_reg, dino_patch,
                       has_mask=has_mask, mask=mask)
            
            logits = out["logits"]
            seg_logits = out["seg_logits"]
            
            cls_loss = criterion_cls(logits, labels) * CFG["cls_loss_weight"]
            total_loss = cls_loss
            
            seg_loss_val = 0.0
            if mask is not None and valid_mask_indices is not None and len(valid_mask_indices) > 0:
                seg_pred = seg_logits[valid_mask_indices]
                seg_gt = mask.unsqueeze(1)
                
                seg_loss = criterion_seg(seg_pred, seg_gt)
                seg_loss_val = seg_loss.item()
                total_loss = total_loss + CFG["seg_loss_weight"] * seg_loss
                
            
                for p, g in zip(seg_pred, seg_gt):
                    iou_sum += iou_score(p.unsqueeze(0), g.unsqueeze(0)).item()
                    iou_count += 1
                seg_samples += len(valid_mask_indices)
                
                
            if "noise_contrast" in out and "dino_contrast" in out:
                contrast_loss = criterion_contrast(out["noise_contrast"], out["dino_contrast"], labels)
                total_loss = total_loss + CFG["contrast_loss_weight"] * contrast_loss
                
                
        
        running_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        running_seg_loss += seg_loss_val * images.size(0)
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        
        if len(visuals) < num_visuals and valid_mask_indices is not None and len(valid_mask_indices) > 0:
            n = min(num_visuals - len(visuals), len(valid_mask_indices))
            vis_indices = valid_mask_indices[:n]
            
            imgs_cpu = images[vis_indices].cpu()
            preds_mask = torch.sigmoid(seg_logits[vis_indices]).cpu()
            gt_mask = mask[:n].unsqueeze(1).cpu()
            
            try:
                img_grid = make_grid(imgs_cpu, nrow=n, normalize=True, scale_each=True)
            except Exception:
                img_grid = make_grid(imgs_cpu, nrow=n, normalize=False)
            
            pred_grid = make_grid(preds_mask.repeat(1, 3, 1, 1), nrow=n, normalize=False)
            gt_grid = make_grid(gt_mask.repeat(1, 3, 1, 1), nrow=n, normalize=False)
            
            visuals.append({"image_grid": img_grid, "pred_grid": pred_grid, "gt_grid": gt_grid})
    
    avg_loss = running_loss / total
    avg_cls = running_cls_loss / total
    avg_seg = (running_seg_loss / total) if seg_samples > 0 else 0.0
    acc = correct / total
    
    preds_cat = torch.cat(all_preds, dim=0).numpy()
    labels_cat = torch.cat(all_labels, dim=0).numpy()
    
    try:
        val_f1_macro = f1_score(labels_cat, preds_cat, average="macro")
        val_f1_micro = f1_score(labels_cat, preds_cat, average="micro")
    except Exception:
        val_f1_macro = compute_macro_f1_torch(torch.tensor(preds_cat), 
                                       torch.tensor(labels_cat), CFG["num_classes"])
        val_f1_micro = 0.0
    
    mean_iou = (iou_sum / iou_count) if iou_count > 0 else 0.0
    
    wandb.log({
        "val/epoch_loss": avg_loss,
        "val/cls_loss": avg_cls,
        "val/seg_loss": avg_seg,
        "val/acc": acc,
        "val/f1_macro": val_f1_macro,
        "val/f1_micro": val_f1_micro,
        "val/iou_masked": mean_iou,
        "val/seg_samples": seg_samples,
        "epoch": epoch
    })
    
    vis_list = []
    for v in visuals:
        vis_list.append(wandb.Image(v["image_grid"].permute(1, 2, 0).numpy(), caption="images"))
        vis_list.append(wandb.Image(v["pred_grid"].permute(1, 2, 0).numpy(), caption="pred_masks"))
        vis_list.append(wandb.Image(v["gt_grid"].permute(1, 2, 0).numpy(), caption="gt_masks"))
    
    if vis_list:
        wandb.log({"val/examples": vis_list, "epoch": epoch})
    
    msg = f"Epoch {epoch} Val - Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1_macro: {val_f1_macro:.4f}, F1_micro: {val_f1_micro:.4f}, IoU: {mean_iou:.4f}"
    log_message(msg)
    
    return avg_loss, acc, val_f1_macro, mean_iou


# ============================================================================
# TRAINING LOOP
# ============================================================================
log_message("="*80)
log_message(f"Starting training for model: {args.model}")
log_message(f"Experiment directory: {EXPERIMENT_DIR}")
log_message("="*80)

best_val_loss = 1e9
best_val_f1 = 0.0

try:
    for epoch in range(1, CFG["epochs"] + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(epoch)
        val_loss, val_acc, val_f1, val_iou = validate(epoch)
        scheduler.step()
        
        wandb.log({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_iou": val_iou
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = CHECKPOINT_DIR / "best.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "cfg": CFG
            }, best_path)
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_val_iou"] = val_iou
            log_message(f"Saved best model at epoch {epoch} with val_loss: {best_val_loss:.4f}")
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_f1_path = CHECKPOINT_DIR / "best_f1.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "cfg": CFG
            }, best_f1_path)
            wandb.run.summary["best_val_f1"] = best_val_f1

    log_message("Training completed successfully!")
    
except Exception as e:
    log_message(f"ERROR during training: {str(e)}")
    import traceback
    log_message(traceback.format_exc())


# ============================================================================
# TESTING PHASE
# ============================================================================
log_message("="*80)
log_message("Starting testing phase on SIDA test set")
log_message("="*80)


# Test dataset definition
class SIDATestDataset(Dataset):
    def __init__(self, embeddings_folder, transform=None):
        self.embeddings_folder = embeddings_folder
        self.transform = transform
        self.TEST_IMG_FOLDERS = {
            "/home/arush/deepfake/sida_net/test_download/test/full_synthetic": 1,
            "/home/arush/deepfake/sida_net/test_download/test/real": 0,
            "/home/arush/deepfake/sida_net/test_download/test/tampered": 2
        }
        self.inverse_label_map = {v: k for k, v in self.TEST_IMG_FOLDERS.items()}
        self.mask_folder = "/home/arush/deepfake/sida_net/test_download/test/masks"
        self.files = [f for f in os.listdir(self.embeddings_folder) if f.endswith('.npz')]
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        embedding_id = self.files[idx]
        embedding_path = os.path.join(self.embeddings_folder, embedding_id)
        
        try:
            embedding = np.load(embedding_path, allow_pickle=True)
        except Exception as e:
            log_message(f"Error loading {embedding_path}: {e}")
            return None

        metadata = embedding['metadata'].item()
        img_id = metadata['img_id']
        label = metadata['label']
        folder = self.inverse_label_map[label]
        
        if label == 0:
            img_name = f"{img_id}.jpg"
        else:
            img_name = f"{img_id}.png"
        
        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert('RGB')
        
        mask_path = os.path.join(self.mask_folder, f"{img_id}_mask.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            mask = None
            
        if self.transform:
            image = self.transform(image)
            
        clip_embed = embedding['clip_layer_embeddings']
        dino_cls = embedding['dino_cls_token']
        dino_reg = embedding['dino_register_tokens']
        dino_patch = embedding['dino_patch_tokens']
        has_mask = metadata['has_mask']
        
        features = {
            'clip_embed': torch.tensor(clip_embed, dtype=torch.float),
            'dino_cls': torch.tensor(dino_cls, dtype=torch.float),
            'dino_reg': torch.tensor(dino_reg, dtype=torch.float),
            'dino_patch': torch.tensor(dino_patch, dtype=torch.float),
            'has_mask': torch.tensor(has_mask, dtype=torch.float),
            'image': image,
        }
        
        return {
            "features": features,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long)
        }


def collate_fn_test(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None

    b_img = []
    b_clip = []
    b_dino_cls = []
    b_dino_reg = []
    b_dino_patch = []
    b_label = []
    b_has_mask = []
    masks = []
    
    for s in batch:
        feats = s["features"]
        b_img.append(feats["image"])
        b_clip.append(feats["clip_embed"])
        b_dino_cls.append(feats["dino_cls"])
        b_dino_reg.append(feats.get("dino_reg", torch.zeros((4, feats["dino_cls"].shape[-1]))))
        b_dino_patch.append(feats["dino_patch"])
        b_label.append(s["label"])
        
        hm = s["features"].get("has_mask", 0)
        if isinstance(hm, torch.Tensor):
            hm_t = (hm.view(-1)[0].item() != 0)
        else:
            hm_t = bool(int(hm))
        b_has_mask.append(torch.tensor(1 if hm_t else 0, dtype=torch.uint8))
        
        raw_mask = s.get("mask", None)
        masks.append(raw_mask)
        
    images = torch.stack(b_img, dim=0)
    clip = torch.stack(b_clip, dim=0)
    dino_cls = torch.stack(b_dino_cls, dim=0)
    dino_reg = torch.stack(b_dino_reg, dim=0)
    dino_patch = torch.stack(b_dino_patch, dim=0)
    labels = torch.stack(b_label).long()
    has_mask = torch.stack(b_has_mask).to(torch.uint8)
    
    if any(m is not None for m in masks):
        proc = []
        for m in masks:
            if m is None:
                proc.append(torch.full((CFG["img_size"], CFG["img_size"]), -1, dtype=torch.int8))
            else:
                mt = _to_mask_tensor(m, CFG["img_size"]).to(torch.int8)
                proc.append(mt)
        masks_t = torch.stack(proc, dim=0)
    else:
        masks_t = None
        
    return {
        "image": images,
        "clip_embed": clip,
        "dino_cls": dino_cls,
        "dino_reg": dino_reg,
        "dino_patch": dino_patch,
        "label": labels,
        "has_mask": has_mask,
        "mask": masks_t
    }


try:
    test_dataset = SIDATestDataset(embeddings_folder=EMBEDDINGS_FOLDER_test, transform=transforms_data)
    test_loader = DataLoader(test_dataset, batch_size=CFG["val_batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"], collate_fn=collate_fn_test, pin_memory=True)

    # Load best model
    best_checkpoint_path = CHECKPOINT_DIR / "best.pth"
    if not best_checkpoint_path.exists():
        log_message(f"ERROR: Best checkpoint not found at {best_checkpoint_path}")
    else:
        log_message(f"Loading best model from {best_checkpoint_path}")
        state_dict = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(state_dict['model_state'])
        model.eval()

        # Test metrics
        all_preds = []
        all_labels = []
        all_iou_scores = []
        seg_samples_count = 0

        if USE_TORCHMETRICS:
            log_message("Initializing torchmetrics on CPU...")
            pixel_f1 = torchmetrics.F1Score(task="binary").to("cpu")
            pixel_auc = torchmetrics.AUROC(task="binary").to("cpu")

        log_message("Starting test evaluation...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if batch is None:
                    continue
                    
                images = batch["image"].to(device)
                clip_embed = batch["clip_embed"].to(device)
                dino_cls = batch["dino_cls"].to(device)
                dino_reg = batch["dino_reg"].to(device)
                dino_patch = batch["dino_patch"].to(device)
                labels = batch["label"].to(device)
                gt_masks_batch = batch["mask"]
                
                out = model(images, clip_embed, dino_cls, dino_reg, dino_patch)
                logits = out["logits"]
                seg_logits = out.get("seg_logits")
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if seg_logits is not None and gt_masks_batch is not None:
                    gt_masks_batch = gt_masks_batch.to(device)
                    valid_mask_idx = (gt_masks_batch[:, 0, 0] != -1)
                    
                    if valid_mask_idx.any():
                        valid_logits = seg_logits[valid_mask_idx]
                        valid_gt = gt_masks_batch[valid_mask_idx].float().unsqueeze(1)
                        
                        seg_samples_count += valid_logits.size(0)
                        
                        ious = batch_iou(valid_logits, valid_gt)
                        all_iou_scores.extend(ious.cpu().numpy())
                        
                        if USE_TORCHMETRICS:
                            valid_gt_int = valid_gt.int()
                            pixel_f1.update(valid_logits.cpu(), valid_gt_int.cpu())
                            pixel_auc.update(valid_logits.cpu(), valid_gt_int.cpu())

        # Classification metrics
        class_names = ['real', 'full_synthetic', 'tampered']
        
        log_message("\n" + "="*80)
        log_message("CLASSIFICATION METRICS")
        log_message("="*80)
        
        report_str = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=8
        )
        log_message(report_str)
        
        cm = confusion_matrix(all_labels, all_preds)
        log_message(f"Confusion Matrix:\n{cm}")
        
        # Per-class metrics
        log_message("\nPer-Class Detailed Metrics:")
        per_class_acc = []
        for i, name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
            class_acc = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            support = tp + fn
            
            per_class_acc.append(class_acc)
            
            msg = f"[{name: <16}] Precision: {precision:.8f}, Recall: {recall:.8f}, F1: {f1:.8f}, Accuracy: {class_acc:.8f}, Support: {support}"
            log_message(msg)
        
        overall_acc = np.trace(cm) / np.sum(cm)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        micro_f1 = f1_score(all_labels, all_preds, average="micro")
        
        log_message(f"\nOverall Accuracy: {overall_acc:.8f}")
        log_message(f"Macro F1-Score: {macro_f1:.8f}")
        log_message(f"Micro F1-Score: {micro_f1:.8f}")
        
        # Segmentation metrics
        log_message("\n" + "="*80)
        log_message("SEGMENTATION METRICS")
        log_message("="*80)
        
        if seg_samples_count > 0:
            mean_iou = np.mean(all_iou_scores)
            log_message(f"Mean IoU (on {seg_samples_count} masked samples): {mean_iou:.8f}")
            
            if USE_TORCHMETRICS:
                log_message("Calculating Pixel-level AUC on CPU...")
                seg_auc = pixel_auc.compute().item()
                log_message(f"Pixel-level AUC: {seg_auc:.8f}")
                
                log_message("Calculating Pixel-level F1-Score on CPU...")
                seg_f1 = pixel_f1.compute().item()
                log_message(f"Pixel-level F1-Score (at 0.5 threshold): {seg_f1:.8f}")
            else:
                log_message("Pixel-level AUC/F1: Skipped (torchmetrics not installed)")
        else:
            log_message("No valid segmentation masks found in test set.")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {args.model}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = PLOTS_DIR / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_message(f"Confusion matrix saved to {cm_path}")
        
        # Save test results
        results = {
            "model": args.model,
            "overall_accuracy": float(overall_acc),
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "per_class_accuracy": {name: float(acc) for name, acc in zip(class_names, per_class_acc)},
            "confusion_matrix": cm.tolist(),
            "classification_report": report_str,
        }
        
        if seg_samples_count > 0:
            results["mean_iou"] = float(mean_iou)
            results["seg_samples_count"] = seg_samples_count
            if USE_TORCHMETRICS:
                results["pixel_auc"] = float(seg_auc)
                results["pixel_f1"] = float(seg_f1)
        
        results_json_path = TEST_RESULTS_DIR / 'test_results.json'
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=4)
        log_message(f"Test results saved to {results_json_path}")
        
        log_message("\nTesting completed successfully!")

except Exception as e:
    log_message(f"ERROR during testing: {str(e)}")
    import traceback
    log_message(traceback.format_exc())

wandb.finish()
log_message("="*80)
log_message("All operations completed!")
log_message("="*80)