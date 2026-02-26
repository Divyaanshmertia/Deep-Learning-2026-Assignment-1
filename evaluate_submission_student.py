"""
Evaluation script for the dataset classification competition (student version).

Loads a saved model (torch.save(model, ...)), runs inference on
validation or test images, and reports overall accuracy.

Usage:
    python evaluate_submission_student.py \
        --model_path  saved_model.pth \
        --model_file  model.py \
        --data_dir    dataset/dataset \
        

import argparse
import csv
import importlib.util
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


#  Constants 
IMG_SIZE   = 224
BATCH_SIZE = 64
NUM_WORKERS = 4


#  Dataset 
class ImageFolderFlat(Dataset):
    """Load images from a flat directory (no sub-folders, no labels)."""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


#  Helpers 
def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def import_model_module(model_file):
    """Dynamically import a student's model file so torch.load can unpickle it."""
    model_file = os.path.abspath(model_file)
    if not os.path.isfile(model_file):
        print(f"ERROR: model file not found: {model_file}")
        sys.exit(1)

    module_dir = os.path.dirname(model_file)
    module_name = os.path.splitext(os.path.basename(model_file))[0]

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, model_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model(model_path, device):
    """Load a full saved model (torch.save(model, ...))."""
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(1)

    model = torch.load(model_path, map_location=device, weights_only=False)

    if not isinstance(model, nn.Module):
        print("ERROR: loaded object is not an nn.Module.")
        print("       Make sure you saved with  torch.save(model, path)")
        print("       NOT  torch.save(model.state_dict(), path)")
        sys.exit(1)

    model.to(device)
    model.eval()
    return model


def read_labels(csv_path):
    """Read a labels CSV  dict keyed by image filename."""
    gt = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            gt[row["image"]] = int(row["label"])
    return gt


#  Inference 
@torch.no_grad()
def predict_flat(model, img_dir, device):
    """Run inference on a flat image directory. Returns {filename: pred_label}."""
    transform = get_eval_transform()
    ds = ImageFolderFlat(img_dir, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    preds = {}
    for images, fnames in loader:
        images = images.to(device)
        outputs = model(images)
        pred_labels = outputs.argmax(dim=1).cpu().tolist()
        for fname, pred in zip(fnames, pred_labels):
            preds[fname] = pred
    return preds


# Main
def evaluate(model_path, model_file, data_dir, split="valid", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Import model definitions
    print(f"Importing model definitions from: {model_file}")
    import_model_module(model_file)

    # 2. Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device)

    # 3. Locate data
    img_dir = os.path.join(data_dir, split)
    labels_csv = os.path.join(data_dir, "valid_labels.csv")

    if not os.path.isdir(img_dir):
        print(f"ERROR: {split}/ directory not found at {img_dir}")
        sys.exit(1)
    if not os.path.isfile(labels_csv):
        print(f"ERROR: valid_labels.csv not found at {labels_csv}")
        sys.exit(1)

    # 4. Read ground truth
    gt = read_labels(labels_csv)

    # 5. Run inference
    print(f"Running inference on {split} images...")
    preds = predict_flat(model, img_dir, device)
    print(f"  {len(preds)} images processed.")

    # 6. Score — only images present in both preds and gt
    correct, total = 0, 0
    for img, pred in preds.items():
        if img in gt:
            correct += int(pred == gt[img])
            total += 1

    if total == 0:
        print("ERROR: no matching images found between predictions and labels.")
        sys.exit(1)

    acc = correct / total * 100
    print(f"\nAccuracy: {acc:.2f}%  ({correct}/{total})")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on the competition dataset."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to saved model (torch.save(model, path))")
    parser.add_argument("--model_file", required=True,
                        help="Path to model.py (contains model class definition)")
    parser.add_argument("--data_dir", required=True,
                        help="Path to dataset dataset root")

    # group.add_argument("--valid", action="store_const", const="valid", dest="split",
    #                    help="Evaluate on validation set")
    args = parser.parse_args()

    evaluate(args.model_path, args.model_file, args.data_dir)
