import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A


class FarmSegDataset(Dataset):
    def __init__(self, root_dir, tokenizer=None, max_length=256, use_english=False, recursive=False, augment=False):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_english = use_english
        self.recursive = recursive
        self.augment = augment

        if self.augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),  
                    translate_percent=0.05,  
                    rotate=(-15, 15),  
                    shear=0,  
                    interpolation=0,  
                    fill=0, 
                    p=0.4 
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussNoise(p=0.1),
                A.GridDropout(ratio=0.15, p=0.15),
            ], additional_targets={'mask': 'mask'})  
        else:
            self.aug = None


        self.image_files = []
        self.img_paths = []
        self.lbl_paths = []
        self.json_paths = []

        if recursive:
            for region in sorted(os.listdir(root_dir)):
                region_path = os.path.join(root_dir, region)
                if not os.path.isdir(region_path):
                    continue
                img_dir = os.path.join(region_path, "img")
                lbl_dir = os.path.join(region_path, "lbl")
                json_dir = os.path.join(region_path, "json")

                if not os.path.exists(img_dir):
                    continue

                imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".tif"))]
                imgs.sort()

                for img_name in imgs:
                    self.image_files.append(img_name)
                    self.img_paths.append(os.path.join(img_dir, img_name))
                    self.lbl_paths.append(os.path.join(lbl_dir, img_name))
                    self.json_paths.append(os.path.join(json_dir, os.path.splitext(img_name)[0] + ".json"))
        else:

            img_dir = os.path.join(root_dir, "img")
            lbl_dir = os.path.join(root_dir, "lbl")
            json_dir = os.path.join(root_dir, "json")

            self.image_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".tif"))]
            self.image_files.sort()

            self.img_paths = [os.path.join(img_dir, f) for f in self.image_files]
            self.lbl_paths = [os.path.join(lbl_dir, f) for f in self.image_files]
            self.json_paths = [os.path.join(json_dir, os.path.splitext(f)[0] + ".json") for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lbl_path = self.lbl_paths[idx]
        json_path = self.json_paths[idx]
        img_name = os.path.basename(img_path)


        image_np = np.array(Image.open(img_path).convert("RGB"))  # [H, W, 3]


        lbl_img = Image.open(lbl_path).convert("RGB")
        lbl_np = np.array(lbl_img).astype(np.uint8)
        r, g, b = lbl_np[:, :, 0], lbl_np[:, :, 1], lbl_np[:, :, 2]
        red_mask = (r > 100) & (r > g + 30) & (r > b + 30)
        mask_np = np.zeros((lbl_np.shape[0], lbl_np.shape[1]), dtype=np.uint8)
        mask_np[red_mask] = 1  # [H, W]


        if self.aug:
            augmented = self.aug(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']


        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0  # [C, H, W], 归一化到 [0,1]
        label = torch.from_numpy(mask_np).unsqueeze(0).float()              # [1, H, W]


        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            text_key = "img_description_eg" if self.use_english else "img_description_cn"
            text = data.get(text_key, "")


        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        else:
            input_ids = text
            attention_mask = None

        return {
            "image": image,
            "label": label,
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "img_name": img_name,
            "img_path": img_path,
            "lbl_path": lbl_path,
            "json_path": json_path
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    img_name = [item["img_name"] for item in batch]
    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "img_name": img_name
    }
