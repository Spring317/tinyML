import time
import os
import argparse
from typing import Tuple, Dict, Any
from collections import Counter
import random

import torch
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms, datasets  # Add this import
from torchvision.datasets import ImageFolder  # Add this specific import

from CustomDataset import CustomDataset
from mcunet.model_zoo import build_model
from utilities import get_device


class BinaryInsectDataset(torch.utils.data.Dataset):
    """
    Binary classification dataset for Insecta.
    Class 0: Dominant class (species with most samples)
    Class 1: All other species (balanced sampling)
    """
    def __init__(self, data_dir, transform=None, train=True, dominant_class=None):
        self.dataset = ImageFolder(root=data_dir, transform=transform)
        self.transform = transform
        self.train = train
        
        # Find the dominant class if not provided
        class_counts = Counter([y for _, y in self.dataset.samples])
        if dominant_class is None:
            self.dominant_class_idx = max(class_counts, key=class_counts.get)
        else:
            self.dominant_class_idx = dominant_class
            
        self.dominant_class_count = class_counts[self.dominant_class_idx]
        
        print(f"Dominant class: {self.dataset.classes[self.dominant_class_idx]} "
              f"(index {self.dominant_class_idx}) with {self.dominant_class_count} samples")
        
        # Create binary classification indices
        self.dominant_indices = []
        self.other_indices = []
        
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label == self.dominant_class_idx:
                self.dominant_indices.append(idx)
            else:
                self.other_indices.append(idx)
        
        # For training, balance the dataset by sampling from other_indices
        if train:
            # Random sample from other_indices to match dominant_class_count
            if len(self.other_indices) > self.dominant_class_count:
                random.seed(42)  # Set seed for reproducibility
                self.other_indices = random.sample(self.other_indices, self.dominant_class_count)
            
            # Combine both class indices for the final dataset
            self.indices = self.dominant_indices + self.other_indices
            random.shuffle(self.indices)
        else:
            # For validation/test, use all samples
            self.indices = self.dominant_indices + self.other_indices
        
        print(f"Dataset size: {len(self.indices)} samples "
              f"({len(self.dominant_indices)} dominant, {len(self.other_indices)} other)")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, original_label = self.dataset[original_idx]
        
        # Convert to binary classification
        binary_label = 0 if original_label == self.dominant_class_idx else 1
        
        return image, binary_label


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch"""
    model.train()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    checked_labels = False
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Tensor guard for binary classification
        if not checked_labels:
            num_classes = model(images).shape[1]
            label_min = labels.min().item()
            label_max = labels.max().item()

            if labels.min() < 0 or labels.max() >= num_classes:
                raise ValueError(
                    f"Invalid labels detected!\n"
                    f"Labels: {labels}\n"
                    f"Min: {label_min}, Max: {label_max}\n"
                    f"Model output classes: {num_classes}"
                )
            checked_labels = True

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(loss=f"{loss.detach().item():.3f}")

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


def train_validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validate the model"""
    model.eval()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Validating", unit="batch", leave=False)
    true_labels, pred_labels = [], []
    
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.detach().item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    return avg_loss, accuracy, float(macro_f1)


def save_model(
    model: torch.nn.Module,
    name: str,
    save_path: str,
    device: torch.device,
    img_size: Tuple[int, int],
):
    """Save the model in PyTorch and ONNX formats"""
    os.makedirs(save_path, exist_ok=True)
    pytorch_path = os.path.join(save_path, f"{name}.pth")
    torch.save(model, pytorch_path)
    print(f"Saved PyTorch model to {pytorch_path}")

    dummy_input = torch.randn(1, 3, *img_size, device=device)
    onnx_path = os.path.join(save_path, f"{name}.onnx")
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {onnx_path}")


def create_binary_insecta_datasets(data_dir: str, img_size: Tuple[int, int], train_split: float = 0.8):
    """Create binary classification datasets for Insecta with proper transforms"""
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the full dataset to find dominant class
    full_dataset = ImageFolder(root=data_dir)
    class_counts = Counter([y for _, y in full_dataset.samples])
    dominant_class_idx = max(class_counts, key=class_counts.get)
    dominant_class_name = full_dataset.classes[dominant_class_idx]
    
    print(f"Found {len(full_dataset.classes)} insect species")
    print(f"Dominant class: {dominant_class_name} with {class_counts[dominant_class_idx]} samples")
    
    # Create binary datasets with transforms
    train_binary_dataset = BinaryInsectDataset(
        data_dir=data_dir,
        transform=train_transform,
        train=True,
        dominant_class=dominant_class_idx
    )
    
    val_binary_dataset = BinaryInsectDataset(
        data_dir=data_dir,
        transform=val_transform,
        train=False,  # Use all samples for validation split
        dominant_class=dominant_class_idx
    )
    
    # Split into train and validation
    total_size = len(train_binary_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, _ = torch.utils.data.random_split(
        train_binary_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Use a portion of val_binary_dataset for validation
    val_total_size = len(val_binary_dataset)
    val_use_size = min(val_size, val_total_size // 5)  # Use 20% of all data for validation
    val_dataset, _ = torch.utils.data.random_split(
        val_binary_dataset, [val_use_size, val_total_size - val_use_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    return train_dataset, val_dataset, dominant_class_name


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments for binary classification training"""
    parser = argparse.ArgumentParser(description='Train binary MCUNet models for Insecta classification')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loader workers (default: 8)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='mcunet-in2', 
                       choices=['mcunet-in1', 'mcunet-in2', 'mcunet-in4', 'mcunet-in5', 'mcunet-in6'],
                       help='MCUNet model variant (default: mcunet-in2)')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/Insecta',
                       help='Path to Insecta dataset directory (default: data/Insecta)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[160, 160],
                       help='Image size for training (height, width) (default: 160 160)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training (default: 0.8)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained models (default: models)')
    
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Training parameters
    BATCH_SIZE = args['batch_size']
    NUM_WORKERS = args['workers']
    NUM_EPOCHS = args['epochs']
    LR = args['lr']
    
    # Model parameters
    MODEL_NAME = args['model']
    IMG_SIZE = tuple(args['img_size'])
    OUTPUT_DIR = args['output_dir']
    
    # Dataset parameters
    DATA_DIR = args['data_dir']
    TRAIN_SPLIT = args['train_split']
    
    print(f"Binary Classification Training for Insecta Dataset")
    print(f"=" * 60)
    print(f"Training parameters:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Learning Rate: {LR}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Train Split: {TRAIN_SPLIT}")
    
    # Create binary classification datasets
    print(f"\nCreating binary classification datasets...")
    train_dataset, val_dataset, dominant_class_name = create_binary_insecta_datasets(
        DATA_DIR, IMG_SIZE, TRAIN_SPLIT
    )
    
    # Save validation dataset for later use
    torch.save(val_dataset, "val_dataset_binary.pt")
    print(f"✓ Validation dataset saved as val_dataset_binary.pt")
    
    NAME = f"{MODEL_NAME}_insecta_binary_{dominant_class_name.replace(' ', '_')}_vs_others"
    print(f"Model name: {NAME}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    
    print(f"✓ Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Build model for binary classification (2 classes)
    print(f"\nBuilding binary classification model: {MODEL_NAME}")
    model, image_size, description = build_model(net_id=MODEL_NAME, pretrained=True)
    in_features = model.classifier.linear.in_features
    model.classifier.linear = torch.nn.Linear(in_features, 2)  # Binary classification
    print(f"✓ Model built with 2 output classes (binary)")
    
    # Set up training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    device = get_device()
    model.to(device)
    print(f"✓ Training setup complete on device: {device}")
    
    # Training loop
    best_acc = -1.0
    best_f1 = -1.0
    print(f"\nStarting binary classification training for {NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, macro_f1 = train_validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"[Epoch {epoch + 1:2d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1 or (macro_f1 == best_f1 and val_acc > best_acc):
            start_save = time.perf_counter()
            best_acc = val_acc
            best_f1 = macro_f1
            save_model(model, f"{NAME}", OUTPUT_DIR, device, IMG_SIZE)
            end_save = time.perf_counter()
            print(f"  ✓ New best model saved! (save time: {end_save - start_save:.2f}s)")
        
        end = time.perf_counter()
        print(f"  Epoch time: {end - start:.2f}s")
    
    print("=" * 60)
    print(f"🎉 Binary classification training completed!")
    print(f"📊 Best accuracy: {best_acc:.4f} with F1-score: {best_f1:.4f}")
    print(f"💾 Model saved as: {NAME}")
    print(f"📁 Validation dataset saved as: val_dataset_binary.pt")
    
    # Print class information
    print(f"\nBinary Classification Setup:")
    print(f"  Class 0: {dominant_class_name} (dominant species)")
    print(f"  Class 1: All other Insecta species (balanced)")