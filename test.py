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
from src.models import FreqDINOv2 , FreqDINOv1
from src.loss import CombinedSegmentationLoss  
import random
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

class SIDADataset(Dataset):
    def __init__(self,  embeddings_folder, transform=None):
        self.embeddings_folder = embeddings_folder
        self.transform = transform
        self.TEST_IMG_FOLDERS = { "/home/arush/deepfake/sida_net/test_download/test/full_synthetic" : 1,  "/home/arush/deepfake/sida_net/test_download/test/real" : 0 , "/home/arush/deepfake/sida_net/test_download/test/tampered" : 2 }
        self.inverse_label_map = {v: k for k, v in self.TEST_IMG_FOLDERS.items()}
        self.mask_folder  = "/home/arush/deepfake/sida_net/test_download/test/masks"
    def __len__(self):
        return len(os.listdir(self.embeddings_folder))
    def __getitem__(self, idx):
        embedding_id = os.listdir(self.embeddings_folder)[idx]
        embedding_path = os.path.join(self.embeddings_folder, embedding_id)
        embedding = np.load(embedding_path, allow_pickle=True)
        metadeta = embedding['metadata'].item()
        img_id = metadeta['img_id']
        label = metadeta['label']
        folder = self.inverse_label_map[label]
        img_name = f"{img_id}.png"
        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.mask_folder, f"{img_id}_mask.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = transforms.ToTensor()(mask)
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
            "label": torch.tensor(label, dtype=torch.float)
        }
        
        
EMBEDDINGS_FOLDER = "/home/arush/deepfake/sida_net/test_download/test/embeddings"
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
    
    
#FOR NOW CLASSIFXATION ONLY

test_dataset = SIDADataset(embeddings_folder=EMBEDDINGS_FOLDER, transform=transforms_data)
test_loader = DataLoader(test_dataset, batch_size=CFG["val_batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], collate_fn=collate_fn, pin_memory=True)



device = torch.device(CFG["device"])
model = FreqDINOv1().to(device)
wts = "/home/arush/deepfake/sida_net/checkpoints_v1/best.pth"

state_dict = torch.load(wts, map_location=device)
model.load_state_dict(state_dict['model_state_dict'])
model.eval()
#Calulate f1 and accuracy classfication reportt
from sklearn.metrics import classification_report, confusion_matrix

all_preds = []
all_labels = [] 
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        images = batch["image"].to(device)
        clip_embed = batch["clip_embed"].to(device)
        dino_cls = batch["dino_cls"].to(device)
        dino_reg = batch["dino_reg"].to(device)
        dino_patch = batch["dino_patch"].to(device)
        labels = batch["label"].to(device)

        logits = model(clip_embed, dino_cls, dino_reg, dino_patch)["logits"]
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
report = classification_report(all_labels, all_preds, target_names=['real', 'full_synthetic', 'tampered'])
cm = confusion_matrix(all_labels, all_preds)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", cm)