import warnings
warnings.filterwarnings("ignore")
import os
import json 
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as T
from sklearn.metrics import f1_score, classification_report, confusion_matrix
try:
    import torchmetrics
    USE_TORCHMETRICS = True
except ImportError:
    print("WARNING: torchmetrics not found. Pixel-level AUC/F1 will not be calculated.")
    print("Please run 'pip install torchmetrics'")
    USE_TORCHMETRICS = False

from src.utils import SIDADataset
from src.models import  FreqDINOv1, FreqDINOv2
from src.loss import CombinedSegmentationLoss  
import random
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms

class SIDADataset(Dataset):
    def __init__(self,  embeddings_folder, transform=None):
        self.embeddings_folder = embeddings_folder
        self.transform = transform
        self.TEST_IMG_FOLDERS = { "/home/arush/deepfake/sida_net/test_download/test/full_synthetic" : 1,  "/home/arush/deepfake/sida_net/test_download/test/real" : 0 , "/home/arush/deepfake/sida_net/test_download/test/tampered" : 2 }
        self.inverse_label_map = {v: k for k, v in self.TEST_IMG_FOLDERS.items()}
        self.mask_folder  = "/home/arush/deepfake/sida_net/test_download/test/masks"
        
    def __len__(self):
        # Handle potential .DS_Store or other hidden files
        self.files = [f for f in os.listdir(self.embeddings_folder) if f.endswith('.npz')]
        return len(self.files)

    def __getitem__(self, idx):
        embedding_id = self.files[idx]
        embedding_path = os.path.join(self.embeddings_folder, embedding_id)
        
        try:
            embedding = np.load(embedding_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {embedding_path}: {e}")
            # Return a dummy item or raise error
            return None # Or handle appropriately

        metadeta = embedding['metadata'].item()
        img_id = metadeta['img_id']
        label = metadeta['label']
        folder = self.inverse_label_map[label]
        if label ==0:
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
        meta_deta = embedding['metadata'].item()
        has_mask = meta_deta['has_mask']
        
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
            "label": torch.tensor(label, dtype=torch.long) # Ensure label is long
        }
        
        
EMBEDDINGS_FOLDER = "/home/arush/deepfake/sida_net/test_download/test/embeddings/test"
CFG = {
    "project": "deepfake-freq-dinov3-fixed",
    "entity": None,
    "epochs": 30,
    "batch_size": 32,
    "val_batch_size": 32,
    "lr": 1e-4,
    "seg_lr": 2e-4,  
    "weight_decay": 1e-4,
    "num_workers": 6,
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
transforms_data = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.ToTensor(),
])

def _to_mask_tensor(m, img_size):
    # Accepts PIL.Image, numpy array, or torch.Tensor -> returns (H,W) torch.uint8 {0,1}
    if m is None:
        return None
    if isinstance(m, Image.Image):
        m_t = TF.to_tensor(m)  # float [0,1], (1,H,W) or (3,H,W)
        if m_t.size(0) > 1:
            m_t = m_t.mean(dim=0, keepdim=True)
        m_t = (m_t.squeeze(0) > 0.5).to(torch.uint8)
        if m_t.shape != (img_size, img_size):
            m_t = TF.resize(m_t.unsqueeze(0), [img_size, img_size], interpolation=InterpolationMode.NEAREST).squeeze(0).to(torch.uint8)
        return m_t
    # numpy array
    if isinstance(m, np.ndarray):
        t = torch.from_numpy(m)
        if t.ndim == 3:
            t = t.mean(dim=2)
        t = (t > 0).to(torch.uint8)
        if t.shape != (img_size, img_size):
            t = TF.resize(t.unsqueeze(0).float(), [img_size, img_size], interpolation=InterpolationMode.NEAREST).squeeze(0).to(torch.uint8)
        return t
    # torch Tensor
    if torch.is_tensor(m):
        t = m
        if t.ndim == 3:
            t = t.mean(dim=0)
        t = (t > 0).to(torch.uint8)
        if t.shape != (img_size, img_size):
            t = TF.resize(t.unsqueeze(0).float(), [img_size, img_size], interpolation=InterpolationMode.NEAREST).squeeze(0).to(torch.uint8)
        return t
    raise TypeError(f"Unsupported mask type: {type(m)}")

def collate_fn(batch):
    # Filter out None items
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
        b_label.append(s["label"]) # Already a tensor
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
                mt = _to_mask_tensor(m, CFG["img_size"]).to(torch.int8) # {0, 1}
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
    
    
def batch_iou(sig_logits, gt_masks, thr=0.5, eps=1e-6):
    """
    Vectorized IoU for a batch.
    sig_logits : (N,1,H,W) raw logits (before sigmoid)
    gt_masks   : (N,1,H,W) binary {0,1}
    Returns    : (N,) IoUs
    """
    pred = (torch.sigmoid(sig_logits) > thr).float()
    gt   = (gt_masks > 0.5).float()
    
    inter = (pred * gt).flatten(1).sum(dim=1)
    union = pred.flatten(1).sum(dim=1) + gt.flatten(1).sum(dim=1) - inter
    iou   = torch.where(union > 0, (inter + eps)/(union + eps), (inter == 0).float())
    return iou
    
    
# --- Dataset and Loader ---
test_dataset = SIDADataset(embeddings_folder=EMBEDDINGS_FOLDER, transform=transforms_data)
test_loader = DataLoader(test_dataset, batch_size=CFG["val_batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], collate_fn=collate_fn, pin_memory=True)

# --- Model Setup ---
device = torch.device(CFG["device"])
model = FreqDINOv2().to(device)
wts = "/home/arush/deepfake/sida_net/checkpoints_fft_phase/best_f1.pth"
state_dict = torch.load(wts, map_location=device)
model.load_state_dict(state_dict['model_state'])
model.eval()

# --- Metrics Initialization ---
all_preds = []
all_labels = [] 
all_iou_scores = []
seg_samples_count = 0

if USE_TORCHMETRICS:
    # +++ CHANGE 1: Initialize on CPU to use RAM instead of VRAM +++
    print("Initializing torchmetrics on CPU to avoid OOM...")
    pixel_f1 = torchmetrics.F1Score(task="binary").to("cpu")
    pixel_auc = torchmetrics.AUROC(task="binary").to("cpu")

# --- Evaluation Loop ---
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
        
        # Classification Metrics
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Segmentation Metrics
        if seg_logits is not None and gt_masks_batch is not None:
            gt_masks_batch = gt_masks_batch.to(device)
            valid_mask_idx = (gt_masks_batch[:, 0, 0] != -1)
            
            if valid_mask_idx.any():
                valid_logits = seg_logits[valid_mask_idx]
                valid_gt = gt_masks_batch[valid_mask_idx].float().unsqueeze(1)
                
                seg_samples_count += valid_logits.size(0)

                # 1. IoU Calculation (still on GPU, which is fast)
                ious = batch_iou(valid_logits, valid_gt)
                all_iou_scores.extend(ious.cpu().numpy())

                # 2. Update pixel-level AUC & F1
                if USE_TORCHMETRICS:
                    valid_gt_int = valid_gt.int() 
                    # +++ CHANGE 2: Move tensors to CPU before updating +++
                    pixel_f1.update(valid_logits.cpu(), valid_gt_int.cpu())
                    pixel_auc.update(valid_logits.cpu(), valid_gt_int.cpu())


# --- Metrics Calculation & Reporting ---

# Create dictionary to store all results
results_data = {
    "classification_metrics": {},
    "segmentation_metrics": {}
}
class_names = ['real', 'full_synthetic', 'tampered']

print("\n--- Classification Metrics ---")
report_str = classification_report(
    all_labels, 
    all_preds, 
    target_names=class_names, 
    digits=8
)
cm = confusion_matrix(all_labels, all_preds)

results_data["classification_metrics"]["report_string"] = report_str
results_data["classification_metrics"]["confusion_matrix"] = cm.tolist()
results_data["classification_metrics"]["per_class"] = {}

print(report_str)
print("Confusion Matrix:\n", cm)

# +++ START: Adapted Per-Class Metrics from your snippet +++
print("\n--- Per-Class Detailed Metrics ---")
per_class_metrics_str = []
for i, name in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    class_metrics = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }
    results_data["classification_metrics"]["per_class"][name] = class_metrics
    
    metric_line = f"[{name: <16}] Precision: {precision:.8f}, Recall: {recall:.8f}, F1-Score: {f1:.8f}"
    print(metric_line)
    per_class_metrics_str.append(metric_line)
# +++ END: Adapted Per-Class Metrics +++


print("\n--- Segmentation Metrics ---")
seg_metrics_str = []
if seg_samples_count > 0:
    # 1. Mean IoU
    mean_iou = np.mean(all_iou_scores)
    results_data["segmentation_metrics"]["mean_iou"] = mean_iou
    results_data["segmentation_metrics"]["masked_samples_count"] = seg_samples_count
    
    line = f"Mean IoU (on {seg_samples_count} masked samples): {mean_iou:.8f}"
    print(line)
    seg_metrics_str.append(line)
    
    if USE_TORCHMETRICS:
        # 2. Pixel-level AUC (This will now run on the CPU)
        print("Calculating Pixel-level AUC on CPU (this may take a moment)...")
        seg_auc = pixel_auc.compute().item()
        results_data["segmentation_metrics"]["pixel_auc"] = seg_auc
        line = f"Pixel-level AUC:                      {seg_auc:.8f}"
        print(line)
        seg_metrics_str.append(line)

        # 3. Pixel-level F1-Score (This will now run on the CPU)
        print("Calculating Pixel-level F1-Score on CPU (this may take a moment)...")
        seg_f1 = pixel_f1.compute().item()
        results_data["segmentation_metrics"]["pixel_f1_score"] = seg_f1
        line = f"Pixel-level F1-Score (at 0.5 thr):    {seg_f1:.8f}"
        print(line)
        seg_metrics_str.append(line)
    else:
        line = "Pixel-level AUC/F1:                   Skipped (torchmetrics not installed)"
        print(line)
        seg_metrics_str.append(line)
else:
    line = "No valid segmentation masks found in the test set."
    print(line)
    seg_metrics_str.append(line)

# --- Save Results to Files ---

SAVE_FILE_name = "test_results.txt"
SAVE_FILE_name_json = "test_results_v1.json"

print(f"\n... Saving human-readable results to 'test_results.txt'")
with open(SAVE_FILE_name, "w") as f:
    f.write("--- Classification Metrics ---\n")
    f.write(report_str)
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array_str(cm))
    f.write("\n\n--- Per-Class Detailed Metrics ---\n")
    f.write("\n".join(per_class_metrics_str))
    f.write("\n\n--- Segmentation Metrics ---\n")
    f.write("\n".join(seg_metrics_str))

