from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from datasets import load_dataset
import random
import numpy as np
import torch
import os

class SIDADataset(Dataset):
    def __init__(self, base_data, embeddings_folder, split='train', transform=None):
        self.base_data = base_data[split]
        self.embeddings_folder = embeddings_folder
        self.transform = transform
    def __len__(self):
        return len(self.base_data)
    def __getitem__(self, idx):
        item = self.base_data[idx]
        image = item['image']
        img_id = item['img_id']
        embedding_path = os.path.join(self.embeddings_folder, img_id + '.npz', )
        embedding = np.load(embedding_path, allow_pickle=True)
        mask = item['mask']
        label = item['label']
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