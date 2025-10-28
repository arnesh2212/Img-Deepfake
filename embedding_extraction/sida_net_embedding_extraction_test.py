# python extract_embeddings.py
# Run with: CUDA_VISIBLE_DEVICES=2 python sida_net_embedding_extraction.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    CLIPModel,
    CLIPImageProcessor,   # <— use the stable (non-fast) processor
    AutoImageProcessor,
    AutoModel,
)
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

# --------------------
# Configuration
# --------------------
# OUTPUT_DIR = Path("/home/arush/deepfake/sida_net/embeddings")
ERROR_LOG = Path("/home/arush/deepfake/sida_net/extraction_errors_test.csv")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HF model ids
CLIP_ID = "openai/clip-vit-base-patch16"
# DINO_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"  # DINOv3 ViT-S/16
DINO_MODEL_ID= "facebook/dinov3-vitl16-pretrain-lvd1689m"
print(f"Using device: {DEVICE}")



# Initialize error log
ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
if not ERROR_LOG.exists():
    with open(ERROR_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'img_id', 'error_message', 'timestamp'])

def log_error(split, img_id, error_msg):
    with open(ERROR_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([split, img_id, str(error_msg), datetime.now().isoformat()])

print("Loading models...")

# --------------------
# CLIP (stable processor)
# --------------------
clip_model = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE)
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_ID)  # <-- fix: non-fast
clip_model.eval()
NUM_CLIP_LAYERS = getattr(clip_model.vision_model.config, "num_hidden_layers", 12)
CLIP_HIDDEN = getattr(clip_model.vision_model.config, "hidden_size", 768)
print(f"CLIP loaded: {CLIP_ID} | layers={NUM_CLIP_LAYERS} | hidden={CLIP_HIDDEN}")

# --------------------
# DINOv3 (HF Transformers)
# --------------------
dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = AutoModel.from_pretrained(DINO_MODEL_ID).to(DEVICE)
dino_model.eval()

# Read config-driven sizes
DINO_PATCH_SIZE = getattr(dino_model.config, "patch_size", 16)
DINO_NUM_REGISTERS = getattr(dino_model.config, "num_register_tokens", 0)
DINO_HIDDEN_SIZE = getattr(dino_model.config, "hidden_size", 384)
DINO_IMAGE_SIZE = getattr(dino_model.config, "image_size", 224)

print(f"DINOv3 loaded: {DINO_MODEL_ID}")
print(f"  Image size: {DINO_IMAGE_SIZE}")
print(f"  Patch size: {DINO_PATCH_SIZE}")
print(f"  Register tokens: {DINO_NUM_REGISTERS}")
print(f"  Hidden size: {DINO_HIDDEN_SIZE}")
print("Models loaded successfully\n")

# --------------------
# Extractors
# --------------------
def extract_clip_embeddings(images):
    """
    Extract CLS-token embeddings from all CLIP vision transformer layers.
    Returns: (batch_size, NUM_CLIP_LAYERS, CLIP_HIDDEN) as np.float32
    """
    try:
        inputs = clip_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)  # (B,3,H,W)

        with torch.inference_mode():
            vision_outputs = clip_model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True
            )
            # Handle both ModelOutput and tuple returns
            hidden_states = (
                vision_outputs.hidden_states
                if hasattr(vision_outputs, "hidden_states")
                else vision_outputs[1]
            )
            layer_embeddings = torch.stack(
                [hidden_states[i][:, 0, :] for i in range(1, NUM_CLIP_LAYERS + 1)],
                dim=1
            )
        return layer_embeddings.detach().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"CLIP extraction error: {e}")
        return None

def extract_dino_embeddings(images):
    """
    DINOv3 via HF:
      returns dict with:
        - cls_token        : (B, C)
        - register_tokens  : (B, n_reg, C) (n_reg may be 0)
        - patch_tokens     : (B, N, C) with N = H_p * W_p
        - concat_tokens    : (B, 1 + n_reg + N, C)
        - patch_grid       : (H_p, W_p)
        - n_reg, hidden
    """
    try:
        inputs = dino_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)  # (B,3,H,W)

        with torch.inference_mode():
            outputs = dino_model(pixel_values=pixel_values, output_hidden_states=False)
            last_hidden = outputs.last_hidden_state  # (B, T, C)

        B, T, C = last_hidden.shape
        n_reg = getattr(dino_model.config, "num_register_tokens", 0)
        _, _, H, W = pixel_values.shape
        H_p, W_p = H // DINO_PATCH_SIZE, W // DINO_PATCH_SIZE
        N = H_p * W_p
        expected_T = 1 + n_reg + N
        if T != expected_T:
            raise ValueError(
                f"Token count mismatch: got {T}, expected {expected_T} "
                f"(1 + {n_reg} + {N}) with H={H}, W={W}, patch={DINO_PATCH_SIZE}"
            )

        cls_token = last_hidden[:, 0, :]                              # (B, C)
        reg_tokens = last_hidden[:, 1:1 + n_reg, :]                   # (B, n_reg, C)
        patch_tokens = last_hidden[:, 1 + n_reg:, :]                  # (B, N, C)

        return {
            "cls_token": cls_token.detach().cpu().numpy().astype(np.float32),
            "register_tokens": reg_tokens.detach().cpu().numpy().astype(np.float32),
            "patch_tokens": patch_tokens.detach().cpu().numpy().astype(np.float32),
            # "concat_tokens": last_hidden.detach().cpu().numpy().astype(np.float32),
            "patch_grid": (H_p, W_p),
            "n_reg": n_reg,
            "hidden": C,
        }

    except Exception as e:
        print(f"DINOv3 extraction error: {e}")
        return None

def create_placeholder_embeddings():
    """Zero placeholders in case a sample fails (keeps shapes consistent)."""
    H_p = W_p = DINO_IMAGE_SIZE // DINO_PATCH_SIZE
    n_patches = H_p * W_p

    return {
        'clip': np.zeros((NUM_CLIP_LAYERS, CLIP_HIDDEN), dtype=np.float32),
        'dino': {
            'cls_token': np.zeros((DINO_HIDDEN_SIZE,), dtype=np.float32),
            'register_tokens': np.zeros((DINO_NUM_REGISTERS, DINO_HIDDEN_SIZE), dtype=np.float32),
            'patch_tokens': np.zeros((n_patches, DINO_HIDDEN_SIZE), dtype=np.float32),
            # 'concat_tokens': np.zeros((1 + DINO_NUM_REGISTERS + n_patches, DINO_HIDDEN_SIZE), dtype=np.float32),
            'patch_grid': (H_p, W_p),
            'n_reg': DINO_NUM_REGISTERS,
        }
    }

# --------------------
# Batch Processing
# --------------------
def process_batch(batch_samples, split):
    """Process a batch of samples and save embeddings (resume + atomic write)."""
    batch_images = []
    batch_metadata = []
    batch_paths = []

    # Prepare batch (skip files that already exist)
    for sample in batch_samples:
        img_id = sample['img_id']
        save_path = OUTPUT_DIR / split / f"{img_id}.npz"
        if save_path.exists():
            continue

        img = sample['image']
        if getattr(img, "mode", None) != 'RGB':
            img = img.convert('RGB')

        batch_images.append(img)
        batch_paths.append(save_path)
        batch_metadata.append({
            'img_id': img_id,
            'label': int(sample['label']),
            'width': int(sample['width']),
            'height': int(sample['height']),
            'has_mask': sample.get('mask') is not None
        })
        # except Exception as e:
        #     log_error(split, sample.get('img_id', 'unknown'), f"Image loading error: {e}")
        #     continue

    if not batch_images:
        return 0

    # Extract embeddings
    clip_embeddings = extract_clip_embeddings(batch_images)
    dino_embeddings = extract_dino_embeddings(batch_images)

    saved_count = 0
    for idx, metadata in enumerate(batch_metadata):
        tmp_path = None
        try:
            save_path = batch_paths[idx]
            # ensure parent exists (defensive)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # temp file ends with .npz so NumPy won't add another extension
            tmp_path = save_path.with_name(save_path.stem + ".__tmp__.npz")

            if clip_embeddings is None or dino_embeddings is None:
                placeholders = create_placeholder_embeddings()
                clip_emb = placeholders['clip']
                dino_emb = placeholders['dino']
                log_error(split, metadata['img_id'], "Extraction failed - using placeholders")
            else:
                clip_emb = clip_embeddings[idx]  # (N_layers, CLIP_HIDDEN)
                dino_emb = {
                    'cls_token': dino_embeddings['cls_token'][idx],           # (C,)
                    'register_tokens': dino_embeddings['register_tokens'][idx], # (n_reg, C) possibly (0,C)
                    'patch_tokens': dino_embeddings['patch_tokens'][idx],     # (N, C)
                    # 'concat_tokens': dino_embeddings['concat_tokens'][idx],   # (1+n_reg+N, C)
                    'patch_grid': dino_embeddings['patch_grid'],
                    'n_reg': dino_embeddings['n_reg'],
                }

            # Tiny JSON header for traceability
            header = {
                "clip_model": CLIP_ID,
                "clip_layers": int(NUM_CLIP_LAYERS),
                "clip_hidden": int(CLIP_HIDDEN),
                "dino_model": DINO_MODEL_ID,
                "dino_patch": int(DINO_PATCH_SIZE),
                "dino_hidden": int(DINO_HIDDEN_SIZE),
                "n_reg": int(dino_emb['n_reg']),
                "patch_grid": list(map(int, dino_emb['patch_grid'])),
            }

            # Atomic write (tmp -> replace)
            np.savez_compressed(
                tmp_path,
                clip_layer_embeddings=clip_emb,
                dino_cls_token=dino_emb['cls_token'],
                dino_register_tokens=dino_emb['register_tokens'],
                dino_patch_tokens=dino_emb['patch_tokens'],
                # dino_concat_tokens=dino_emb['concat_tokens'],
                dino_spatial_dims=np.array(dino_emb['patch_grid'], dtype=np.int32),
                dino_n_reg=np.int32(dino_emb['n_reg']),
                metadata=metadata,
                header=json.dumps(header),
            )
            os.replace(tmp_path, save_path)
            saved_count += 1

        except Exception as e:
            log_error(split, metadata['img_id'], f"Save error: {e}")
            try:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    return saved_count

# # --------------------
# # Main
# # --------------------
# def main():
    
#     # from datasets import load_from_disk, DatasetDict
#     # from pathlib import Path

#     # LOCAL_DS_DIR = Path("/home/arush/deepfake/sida_net/datasets/SID_Set")

#     # print("Loading dataset...")
#     # if LOCAL_DS_DIR.exists():
#     #     # Load locally, fully bypassing HF datasets cache & network
#     #     ds = load_from_disk(str(LOCAL_DS_DIR))
#     # else:
#     #     # First-time download, then persist to disk
#     #     #Create folder
#     #     if os.path.exists(LOCAL_DS_DIR)==False:
#     #         os.makedirs(LOCAL_DS_DIR)
#     #     ds = load_dataset("saberzl/SID_Set")
#     #     LOCAL_DS_DIR.parent.mkdir(parents=True, exist_ok=True)
#     #     ds.save_to_disk(str(LOCAL_DS_DIR))
    
#     print("Loading dataset...")
#     ds = load_dataset("saberzl/SID_Set")



#     for split in ['train', 'validation']:
#         print(f"\n{'='*60}")
#         print(f"Processing {split.upper()} split")
#         print(f"{'='*60}")

#         dataset = ds[split]
#         n_samples = len(dataset)
#         n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE

#         total_saved = 0
#         for batch_idx in tqdm(range(n_batches), desc=f"Extracting {split}"):
#             start_idx = batch_idx * BATCH_SIZE
#             end_idx = min(start_idx + BATCH_SIZE, n_samples)
#             batch_samples = [dataset[i] for i in range(start_idx, end_idx)]
#             total_saved += process_batch(batch_samples, split)

#         already = len(list((OUTPUT_DIR / split).glob("*.npz"))) - total_saved
#         print(f"\nCompleted {split} split:")
#         print(f"  Total samples: {n_samples}")
#         print(f"  New embeddings saved: {total_saved}")
#         print(f"  Already existed (approx): {already}")

#     print(f"\n{'='*60}")
#     print("Extraction complete!")
#     print(f"Embeddings saved to: {OUTPUT_DIR}")
#     print(f"Error log saved to: {ERROR_LOG}")
#     print(f"{'='*60}")

# if __name__ == "__main__":
    
#     main()


TEST_IMG_FOLDERS = { "/home/arush/deepfake/sida_net/test_download/test/full_synthetic" : 1,  "/home/arush/deepfake/sida_net/test_download/test/real" : 0 , "/home/arush/deepfake/sida_net/test_download/test/tampered" : 2 }
OUTPUT_DIR = Path("/home/arush/deepfake/sida_net/test_download/test/embeddings")
MASK_FOLDER  = "/home/arush/deepfake/sida_net/test_download/test/masks"
from tqdm.auto import tqdm
from PIL import Image
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for folder in TEST_IMG_FOLDERS.keys():
    
    batch_size = 32
    img_list = os.listdir(folder)
    n_samples = len(img_list)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    total_saved = 0
    for batch_idx in tqdm(range(n_batches), desc=f"Extracting {folder}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_samples = []
        for i in range(start_idx, end_idx):
            img = img_list[i]
            
            img_path = os.path.join(folder,img)
            image = Image.open(img_path).convert('RGB')
            label = TEST_IMG_FOLDERS[folder]
            width, height = image.size
            mask_img_name = img.split('.')[0] + '_mask.png'
            mask_path = os.path.join(MASK_FOLDER,mask_img_name)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                mask = None

            sample = {
                'img_id': img.split('.')[0],
                'image': image,
                'label': label,
                'width': width, 
                'height': height,
                'mask': mask
            }
            batch_samples.append(sample)
        
        total_saved += process_batch(batch_samples, 'test')