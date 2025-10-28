import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from src.utils import SIDADataset
from src.models import FreqDINOv1
import random
from tqdm import tqdm
from datasets import load_dataset
_HAS_SKLEARN = True
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from datasets import load_from_disk, DatasetDict
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from PIL import Image

LOCAL_DS_DIR = Path("/home/arush/deepfake/sida_net/datasets/SID_Set")

CFG = {
    "project": "deepfake-freq-dinov3",
    "entity": None,
    "epochs": 30,
    "batch_size": 32,
    "val_batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 6,
    "log_interval": 50,
    "img_size": 224,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seg_loss_weight": 0.5,
    "cls_loss_weight": 1.0,
    "checkpoint_dir": "./checkpoints_v1",
    "project_tags": ["freq", "dino", "clip", "segmentation"],
}
os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(CFG["seed"])


def _to_mask_tensor(m, img_size):
    # Accepts PIL.Image, numpy array, or torch.Tensor -> returns (H,W) torch.uint8 {0,1}
    if m is None:
        return None
    if isinstance(m, Image.Image):
        m_t = TF.to_tensor(m)  # float [0,1], (1,H,W) or (3,H,W)
        # if rgb mask -> take first channel; if float mask threshold
        if m_t.size(0) > 1:
            m_t = m_t.mean(dim=0, keepdim=True)
        m_t = (m_t.squeeze(0) > 0.5).to(torch.uint8)
        if m_t.shape != (img_size, img_size):
            m_t = TF.resize(m_t.unsqueeze(0), [img_size, img_size]).squeeze(0).to(torch.uint8)
        return m_t
    # numpy array
    if isinstance(m, np.ndarray):
        t = torch.from_numpy(m)
        if t.ndim == 3:
            t = t.mean(dim=2)
        t = (t > 0).to(torch.uint8)
        if t.shape != (img_size, img_size):
            t = TF.resize(t.unsqueeze(0).float(), [img_size, img_size]).squeeze(0).to(torch.uint8)
        return t
    # torch Tensor
    if torch.is_tensor(m):
        t = m
        if t.ndim == 3:
            t = t.mean(dim=0)
        t = (t > 0).to(torch.uint8)
        if t.shape != (img_size, img_size):
            t = TF.resize(t.unsqueeze(0).float(), [img_size, img_size]).squeeze(0).to(torch.uint8)
        return t
    raise TypeError(f"Unsupported mask type: {type(m)}")

def collate_fn(batch):
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
        b_label.append(torch.tensor(s["label"], dtype=torch.long))

        # normalize has_mask to 0/1 tensor
        hm = s.get("has_mask", 0)
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

    # process masks: convert each to tensor or placeholder (-1)
    if any(m is not None for m in masks):
        proc = []
        for m in masks:
            if m is None:
                # placeholder filled with -1 so you can detect unavailable masks later
                proc.append(torch.full((CFG["img_size"], CFG["img_size"]), -1, dtype=torch.int8))
            else:
                mt = _to_mask_tensor(m, CFG["img_size"]).to(torch.int8)
                proc.append(mt)
        masks_t = torch.stack(proc, dim=0)  # (B, H, W), dtype int8
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


def iou_score(pred_mask, true_mask, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_mask) > thr).float()
    inter = (pred * true_mask).sum()
    union = pred.sum() + true_mask.sum() - inter
    if union.item() == 0:
        return torch.tensor(1.0) if inter.item() == 0 else torch.tensor(0.0)
    return (inter + eps) / (union + eps)

BASE_DATA = load_from_disk(str(LOCAL_DS_DIR))

EMBEDDINGS_FOLDER_train = "/home/arush/deepfake/sida_net/embeddings/train"
EMBEDDINGS_FOLDER_val = "/home/arush/deepfake/sida_net/embeddings/validation"


transforms_data = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.ToTensor(),
])



train_ds = SIDADataset(BASE_DATA, EMBEDDINGS_FOLDER_train, split='train',
                    transform= transforms_data)
val_ds = SIDADataset(BASE_DATA, EMBEDDINGS_FOLDER_val, split='validation',
                    transform= transforms_data)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                        num_workers=CFG["num_workers"], collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CFG["val_batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], collate_fn=collate_fn, pin_memory=True)

device = torch.device(CFG["device"])
model = FreqDINOv1().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
criterion = nn.CrossEntropyLoss()

wandb_mode = os.getenv("WANDB_MODE", "online")
wandb.init(project=CFG["project"], entity=CFG["entity"], config=CFG, tags=CFG["project_tags"], mode=wandb_mode)
wandb.watch(model, log="all", log_freq=100)

# fallback F1 if sklearn not available
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

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        images = batch["image"].to(device)
        clip_embed = batch["clip_embed"].to(device)
        dino_cls = batch["dino_cls"].to(device)
        dino_reg = batch["dino_reg"].to(device)
        dino_patch = batch["dino_patch"].to(device)
        labels = batch["label"].to(device)
        has_mask = batch["has_mask"].to(device) if batch["has_mask"] is not None else None
        mask = batch["mask"].to(device) if batch["mask"] is not None else None

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            out = model(images, clip_embed, dino_cls, dino_reg, dino_patch, has_mask=has_mask, mask=mask)
            logits = out["logits"]
            seg_loss = out["seg_loss"]
            cls_loss = criterion(logits, labels) * CFG["cls_loss_weight"]
            total_loss = cls_loss
            if seg_loss is not None:
                total_loss = total_loss + CFG["seg_loss_weight"] * seg_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        if seg_loss is not None:
            running_seg_loss += seg_loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if (step + 1) % CFG["log_interval"] == 0:
            wandb.log({
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step_loss": total_loss.item(),
                "train/cls_loss": cls_loss.item(),
                "train/seg_loss": seg_loss.item() if seg_loss is not None else 0.0,
                "epoch": epoch,
                "step": epoch * len(train_loader) + step
            })

    avg_loss = running_loss / total
    avg_cls = running_cls_loss / total
    avg_seg = (running_seg_loss / total) if running_seg_loss > 0 else 0.0
    acc = correct / total

    preds_cat = torch.cat(all_preds, dim=0).numpy()
    labels_cat = torch.cat(all_labels, dim=0).numpy()
    if _HAS_SKLEARN:
        try:
            train_f1 = f1_score(labels_cat, preds_cat, average="macro")
        except Exception:
            train_f1 = compute_macro_f1_torch(torch.tensor(preds_cat), torch.tensor(labels_cat), CFG["num_classes"])
    else:
        train_f1 = compute_macro_f1_torch(torch.tensor(preds_cat), torch.tensor(labels_cat), CFG["num_classes"])

    wandb.log({
        "train/epoch_loss": avg_loss,
        "train/cls_loss_epoch": avg_cls,
        "train/seg_loss_epoch": avg_seg,
        "train/acc": acc,
        "train/f1": train_f1,
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
    for step, batch in  enumerate(tqdm(val_loader, desc=f"Val Epoch {epoch}")):
        images = batch["image"].to(device)
        clip_embed = batch["clip_embed"].to(device)
        dino_cls = batch["dino_cls"].to(device)
        dino_reg = batch["dino_reg"].to(device)
        dino_patch = batch["dino_patch"].to(device)
        labels = batch["label"].to(device)
        has_mask = batch["has_mask"].to(device) if batch["has_mask"] is not None else None
        mask = batch["mask"].to(device) if batch["mask"] is not None else None

        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            out = model(images, clip_embed, dino_cls, dino_reg, dino_patch, has_mask=has_mask, mask=mask)
            logits = out["logits"]
            seg_loss = out["seg_loss"]
            cls_loss = criterion(logits, labels) * CFG["cls_loss_weight"]
            total_loss = cls_loss
            if seg_loss is not None:
                total_loss = total_loss + CFG["seg_loss_weight"] * seg_loss

        running_loss += total_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        if seg_loss is not None:
            running_seg_loss += seg_loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        if (mask is not None) and (has_mask is not None):
            has_mask_bool = has_mask.view(-1).bool()
            if has_mask_bool.any():
                seg_pred = out["seg_logits"][has_mask_bool]
                seg_gt = mask[has_mask_bool].unsqueeze(1).float()
                for p, g in zip(seg_pred, seg_gt):
                    iou_sum += iou_score(p.unsqueeze(0), g.unsqueeze(0)).item()
                    iou_count += 1

        if len(visuals) < num_visuals:
            n = min(num_visuals - len(visuals), images.size(0))
            imgs_cpu = images[:n].cpu()  # float [0,1], (N,3,H,W)

            # preds_mask: (N,1,H,W), float in [0,1]
            preds_mask = torch.sigmoid(out["seg_logits"][:n].cpu())

            # gt_mask may contain -1 placeholders (int8). convert to float and set unavailable -> 0
            if mask is not None:
                gt_mask_raw = mask[:n].cpu()  # (N,H,W), dtype may be int8
                # convert to float and map -1 -> 0
                gt_mask = gt_mask_raw.clone().float()
                gt_mask[gt_mask_raw == -1] = 0.0
                gt_mask = gt_mask.unsqueeze(1)  # (N,1,H,W)
            else:
                gt_mask = torch.zeros((n, 1, CFG["img_size"], CFG["img_size"]), dtype=torch.float32)

            # image grid: normalize images for display; masks: don't normalize (already 0..1)
            try:
                img_grid = make_grid(imgs_cpu, nrow=n, normalize=True, scale_each=True)
            except Exception:
                img_grid = make_grid(imgs_cpu, nrow=n, normalize=False)

            pred_grid = make_grid(preds_mask.repeat(1,3,1,1), nrow=n, normalize=False)
            gt_grid = make_grid(gt_mask.repeat(1,3,1,1), nrow=n, normalize=False)

            visuals.append({"image_grid": img_grid, "pred_grid": pred_grid, "gt_grid": gt_grid})


    avg_loss = running_loss / total
    avg_cls = running_cls_loss / total
    avg_seg = (running_seg_loss / total) if running_seg_loss > 0 else 0.0
    acc = correct / total

    preds_cat = torch.cat(all_preds, dim=0).numpy()
    labels_cat = torch.cat(all_labels, dim=0).numpy()
    if _HAS_SKLEARN:
        try:
            val_f1 = f1_score(labels_cat, preds_cat, average="macro")
        except Exception:
            val_f1 = compute_macro_f1_torch(torch.tensor(preds_cat), torch.tensor(labels_cat), CFG["num_classes"])
    else:
        val_f1 = compute_macro_f1_torch(torch.tensor(preds_cat), torch.tensor(labels_cat), CFG["num_classes"])

    mean_iou = (iou_sum / iou_count) if iou_count > 0 else float("nan")

    wandb.log({
        "val/epoch_loss": avg_loss,
        "val/cls_loss": avg_cls,
        "val/seg_loss": avg_seg,
        "val/acc": acc,
        "val/f1": val_f1,
        "val/iou_masked": mean_iou,
        "epoch": epoch
    })

    vis_list = []
    for v in visuals:
        vis_list.append(wandb.Image(v["image_grid"].permute(1,2,0).numpy(), caption="images"))
        vis_list.append(wandb.Image(v["pred_grid"].permute(1,2,0).numpy(), caption="pred_masks"))
        vis_list.append(wandb.Image(v["gt_grid"].permute(1,2,0).numpy(), caption="gt_masks"))
    if vis_list:
        wandb.log({"val/examples": vis_list, "epoch": epoch})

    return avg_loss, acc, val_f1, mean_iou

best_val_loss = 1e9
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

    ckpt_path = os.path.join(CFG["checkpoint_dir"], f"epoch{epoch:03d}_val{val_loss:.4f}.pth")
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

wandb.finish()
