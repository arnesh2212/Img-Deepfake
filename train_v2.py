import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as T
from sklearn.metrics import f1_score
from src.utils import SIDADataset
from src.models import FreqDINOv2  
from src.loss import CombinedSegmentationLoss  
import random
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
LOCAL_DS_DIR = Path("/home/arush/deepfake/sida_net/datasets/SID_Set")


CFG = {
    "project": "deepfake-freq-dinov3-final",
    "entity": None,
    "epochs": 30,
    "batch_size": 32,
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
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "checkpoint_dir": "./checkpoints_v2",
    "project_tags": ["freq", "dino", "clip", "segmentation", "fixed"],
    "num_classes": 3,
}
os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])

#     return None
def _to_mask_tensor(m, img_size):
    if m is None:
        return None

    def _resize_nn(x):
        # expects (1,H,W) float tensor in [0,1]
        if x.shape[-2:] != (img_size, img_size):
            x = TF.resize(x, [img_size, img_size], interpolation=InterpolationMode.NEAREST)
        return x

    if isinstance(m, Image.Image):
        # PIL -> [0,1] float CxHxW
        t = TF.to_tensor(m)
        if t.size(0) > 1:
            t = t.mean(dim=0, keepdim=True)  # 1xHxW
        else:
            t = t  # 1xHxW
        t = _resize_nn(t)
        t = (t > 0.5).float()
        return t.squeeze(0)  # HxW

    if isinstance(m, np.ndarray):
        # np -> float [0,1], handle 0/255
        t = torch.from_numpy(m).float()
        if t.ndim == 3:
            t = t.mean(dim=2)  # HxW
        # normalize if looks like 0/255
        if t.max() > 1.0:
            t = t / 255.0
        t = t.clamp(0,1).unsqueeze(0)  # 1xHxW
        t = _resize_nn(t)
        t = (t > 0.5).float()
        return t.squeeze(0)

    if torch.is_tensor(m):
        t = m.float()
        if t.ndim == 3:
            t = t.mean(dim=0)  # HxW
        if t.max() > 1.0:
            t = t / 255.0
        t = t.clamp(0,1).unsqueeze(0)  # 1xHxW
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
        # keep dtype consistent with feat tensors
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

    # has_mask: 1 if mask exists, else 0
    has_mask = torch.tensor([1 if m is not None else 0 for m in masks], dtype=torch.uint8)

    valid_mask_indices = []
    processed_masks = []
    
    for i, m in enumerate(masks):
        mt = _to_mask_tensor(m, CFG["img_size"])
        if mt is not None and has_mask[i].item() == 1:
            valid_mask_indices.append(i)
            processed_masks.append(mt)
    
    # Create mask tensor only for samples with valid masks
    if len(processed_masks) > 0:
        masks_t = torch.stack(processed_masks, dim=0)  # (N_valid, H, W)
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
    """Compute IoU between predicted and true masks"""
    pred = (torch.sigmoid(pred_mask) > thr).float()
    inter = (pred * true_mask).sum()
    union = pred.sum() + true_mask.sum() - inter
    if union.item() == 0:
        return torch.tensor(1.0) if inter.item() == 0 else torch.tensor(0.0)
    return (inter + eps) / (union + eps)

def batch_iou(sig_logits, gt_masks, thr=0.5, eps=1e-6):
    """
    Vectorized IoU for a batch.
    sig_logits : (N,1,H,W) raw logits (before sigmoid)
    gt_masks   : (N,1,H,W) binary {0,1}
    Returns    : (N,) IoUs
    CHANGES:
    - Ensures consistent pairing and correct empty-union cases:
      IoU==1 if both pred and GT are empty, else 0 if only one is empty.
    """
    pred = (torch.sigmoid(sig_logits) > thr).float()
    gt   = (gt_masks > 0.5).float()

    inter = (pred * gt).flatten(1).sum(dim=1)
    union = pred.flatten(1).sum(dim=1) + gt.flatten(1).sum(dim=1) - inter
    iou   = torch.where(union > 0, (inter + eps)/(union + eps), (inter == 0).float())
    return iou

# Load datasets
BASE_DATA = load_from_disk(str(LOCAL_DS_DIR))


EMBEDDINGS_FOLDER_train = "/home/arush/deepfake/sida_net/embeddings/train"
EMBEDDINGS_FOLDER_val = "/home/arush/deepfake/sida_net/embeddings/validation"

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

# Initialize model and losses
device = torch.device(CFG["device"])
model = FreqDINOv2(num_classes=CFG["num_classes"]).to(device)

# Separate optimizers for classification and segmentation
cls_params = [p for n, p in model.named_parameters() if 'seg_decoder' not in n]
seg_params = [p for n, p in model.named_parameters() if 'seg_decoder' in n]

optimizer = torch.optim.AdamW([
    {'params': cls_params, 'lr': CFG["lr"]},
    {'params': seg_params, 'lr': CFG["seg_lr"]}  # Higher LR for seg head
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

# WandB
wandb_mode = os.getenv("WANDB_MODE", "online")
wandb.init(project=CFG["project"], entity=CFG["entity"], config=CFG, 
           tags=CFG["project_tags"], mode=wandb_mode)
wandb.watch(model, log="all", log_freq=100)


def compute_macro_f1_torch(preds, targets, num_classes):
    """Fallback F1 computation"""
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


def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
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
            
            # Classification loss
            cls_loss = criterion_cls(logits, labels) * CFG["cls_loss_weight"]
            total_loss = cls_loss
            
            # Segmentation loss (only for samples with valid masks)
            seg_loss_val = 0.0
            if mask is not None and valid_mask_indices is not None and len(valid_mask_indices) > 0:
                # Extract predictions for samples with valid masks
                seg_pred = seg_logits[valid_mask_indices]
                seg_gt = mask.unsqueeze(1)  # (N_valid, 1, H, W)
                
                # Compute segmentation loss
                seg_loss = criterion_seg(seg_pred, seg_gt)
                seg_loss_val = seg_loss.item()
                total_loss = total_loss + CFG["seg_loss_weight"] * seg_loss
                seg_samples += len(valid_mask_indices)
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        running_seg_loss += seg_loss_val * images.size(0)
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        if (step + 1) % CFG["log_interval"] == 0:
            wandb.log({
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/seg_lr": optimizer.param_groups[1]["lr"],
                "train/step_loss": total_loss.item(),
                "train/cls_loss": cls_loss.item(),
                "train/seg_loss": seg_loss_val,
                "epoch": epoch,
                "step": epoch * len(train_loader) + step
            })
    
    avg_loss = running_loss / total
    avg_cls = running_cls_loss / total
    avg_seg = (running_seg_loss / total) if seg_samples > 0 else 0.0
    acc = correct / total
    
    preds_cat = torch.cat(all_preds, dim=0).numpy()
    labels_cat = torch.cat(all_labels, dim=0).numpy()
    
    try:
        train_f1 = f1_score(labels_cat, preds_cat, average="macro")
    except Exception:
        train_f1 = compute_macro_f1_torch(torch.tensor(preds_cat), 
                                         torch.tensor(labels_cat), CFG["num_classes"])
    
    wandb.log({
        "train/epoch_loss": avg_loss,
        "train/cls_loss_epoch": avg_cls,
        "train/seg_loss_epoch": avg_seg,
        "train/acc": acc,
        "train/f1": train_f1,
        "train/seg_samples": seg_samples,
        "epoch": epoch
    })
    
    return avg_loss, acc, train_f1


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
                
                # Compute IoU
                for p, g in zip(seg_pred, seg_gt):
                    iou_sum += iou_score(p.unsqueeze(0), g.unsqueeze(0)).item()
                    iou_count += 1
                
                seg_samples += len(valid_mask_indices)
        
        running_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        running_seg_loss += seg_loss_val * images.size(0)
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        

        
        # Visualizations
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
        val_f1 = f1_score(labels_cat, preds_cat, average="macro")
    except Exception:
        val_f1 = compute_macro_f1_torch(torch.tensor(preds_cat), 
                                       torch.tensor(labels_cat), CFG["num_classes"])
    
    mean_iou = (iou_sum / iou_count) if iou_count > 0 else 0.0
    
    wandb.log({
        "val/epoch_loss": avg_loss,
        "val/cls_loss": avg_cls,
        "val/seg_loss": avg_seg,
        "val/acc": acc,
        "val/f1": val_f1,
        "val/iou_masked": mean_iou,
        "val/seg_samples": seg_samples,
        "epoch": epoch
    })
    
    # Log visualizations
    vis_list = []
    for v in visuals:
        vis_list.append(wandb.Image(v["image_grid"].permute(1, 2, 0).numpy(), caption="images"))
        vis_list.append(wandb.Image(v["pred_grid"].permute(1, 2, 0).numpy(), caption="pred_masks"))
        vis_list.append(wandb.Image(v["gt_grid"].permute(1, 2, 0).numpy(), caption="gt_masks"))
    
    if vis_list:
        wandb.log({"val/examples": vis_list, "epoch": epoch})
    
    return avg_loss, acc, val_f1, mean_iou



# Training loop
best_val_loss = 1e9
best_val_f1 = 0.0
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
    
    # Save checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = os.path.join(CFG["checkpoint_dir"], "best.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "cfg": CFG
        }, best_path)
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_val_iou"] = val_iou
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_f1_path = os.path.join(CFG["checkpoint_dir"], "best_f1.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "cfg": CFG
        }, best_f1_path)
        wandb.run.summary["best_val_f1"] = best_val_f1

wandb.finish()