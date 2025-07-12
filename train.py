import time
import os
import argparse
from typing import Tuple, Dict, Any

import torch
from sklearn.metrics import (
    f1_score,
)
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomDataset import CustomDataset
from mcunet.model_zoo import build_model
from utilities import get_device, manifest_generator_wrapper
from data_prep.create_dataset import DatasetCreator


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[CustomDataset],
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
        Trains the given model for one epoch on the provided DataLoader.

        This function performs standard supervised learning with forward pass,
        loss computation, backpropagation, and optimizer updates. It also includes
        a label sanity check to ensure labels fall within valid class index range.
    CLASS_NAME
        Args:
            model (torch.nn.Module):
                The model to be trained.
            dataloader (DataLoader[CustomDataset]):
                A DataLoader that yields batches of (image, label) pairs.
            criterion:
                A loss function (e.g., nn.CrossEntropyLoss).
            optimizer:
                A PyTorch optimizer (e.g., torch.optim.Adam or SGD).
            device (torch.device):
                The device on which to perform computation (CPU or GPU).

        Returns:
            Tuple[float, float]:
                A tuple containing:
                    - The average training loss over the entire dataset.
                    - The training accuracy over the entire dataset.

        Notes:
            - A tensor guard checks that the ground-truth labels fall within the valid range `[0, num_classes - 1]` based on model output shape.
            - Accumulates total correct predictions and total loss for final reporting.
    """
    model.train()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    checked_labels = False
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # tensor guard
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

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    return avg_loss, accuracy


def train_validate(
    model: torch.nn.Module,
    dataloader: DataLoader[CustomDataset],
    criterion,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Trains the model for one epoch with sparsity regularization applied via a pruner.

    This function is used during a warm-up phase of sparse training, where a regularizer (e.g., L1/L2 penalty on weights or activations) is applied to encourage sparsity before actual pruning is performed.

    Args:
        model (torch.nn.Module):
            The model to be trained.
        dataloader (DataLoader[CustomDataset]):
            A DataLoader that yields batches of (image, label) pairs.
        criterion:
            The loss function used to train the model (e.g., CrossEntropyLoss).
        pruner:
            A sparsity regularizer object that provides:
                - `update_regularizer()`: Called once before training.
                - `regularize(model)`: Called on each backward pass to apply regularization.
        device (torch.device):
            The device on which to perform training (CPU or GPU).

    Returns:
        Tuple[float, float]:
            A tuple containing:
                - Average loss over the epoch.
                - Accuracy over the entire dataset.
    """
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

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    return avg_loss, accuracy, float(macro_f1)


def save_model(
    model: torch.nn.Module,
    name: str,
    save_path: str,
    device: torch.device,
    img_size: Tuple[int, int],
):
    """
    Saves a PyTorch model in both `.pth` and ONNX formats.

    This function exports the given model to:
        - PyTorch format (.pth) using `torch.save()`
        - ONNX format (.onnx) using `torch.onnx.export()`, with support for dynamic batch sizes

    Args:
        model (torch.nn.Module):
            The trained PyTorch model to be saved.
        name (str):
            Base name for the output files (e.g., 'mobilenetv3').
        save_path (str):
            Directory where the model files will be saved. Will be created if it doesn't exist.
        device (torch.device):
            Device on which to create the dummy input tensor for ONNX export.
        img_size (Tuple[int, int]):
            Expected input image size as (height, width) for dummy input.

    Output Files:
        - `<save_path>/<name>.pth`: PyTorch serialized model.
        - `<save_path>/<name>.onnx`: ONNX exported model with dynamic batch dimension.

    Notes:
        - ONNX export uses `opset_version=14` and includes constant folding for optimization.
        - Assumes the model expects input shape `(N, 3, H, W)` where H and W are from `img_size`.
        - Dynamic axes allow for variable batch sizes during ONNX inference.
    """
    os.makedirs(save_path, exist_ok=True)
    pytorch_path = os.path.join(save_path, f"{name}.pth")
    torch.save(model, pytorch_path)
    print(f"Saved Pytorch model to {pytorch_path}")

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


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments for the training script.

    Returns:
        Dict[str, Any]: Dictionary containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train MCUNet models for TinyML applications"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers (default: 8)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="mcunet-in2",
        choices=["mcunet-in1", "mcunet-in2", "mcunet-in4", "mcunet-in5", "mcunet-in6"],
        help="MCUNet model variant (default: mcunet-in2)",
    )

    # Dataset parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Dominance threshold for dataset selection (default: 0.5)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[160, 160],
        help="Image size for training (height, width) (default: 160 160)",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)",
    )

    parser.add_argument(
        "--start_rank",
        type=int,
        default=0,
        help="Starting rank for distributed training (default: 0)",
    )
    # Parse the arguments
    args = parser.parse_args()
    return vars(args)  # Convert to dictionary


# Now replace your hardcoded parameters with the command-line arguments
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Training parameters
    BATCH_SIZE = args["batch_size"]
    NUM_WORKERS = args["workers"]
    NUM_EPOCHS = args["epochs"]
    LR = args["lr"]

    # Model parameters
    MODEL_NAME = args["model"]
    IMG_SIZE = tuple(args["img_size"])
    OUTPUT_DIR = args["output_dir"]

    # Dataset parameters
    DOMINANCE_THRESHOLD = args["threshold"]
    START_RANK = args["start_rank"]

    print("Training with parameters:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Learning Rate: {LR}")
    print(f"  Dominance Threshold: {DOMINANCE_THRESHOLD}")

    # Load dataset with threshold
    _, _, _, species_labels, _ = manifest_generator_wrapper(
        DOMINANCE_THRESHOLD, export=True
    )
    NUM_SPECIES = len(species_labels.keys())

    print(f"Number of species from manifest: {NUM_SPECIES}")
    datacreator = DatasetCreator(number_of_dominant_classes=NUM_SPECIES)

    # The actual number of classes in the dataset is NUM_SPECIES + 1 (including "Other" class)

    NAME = f"{MODEL_NAME}_haute_garonne_{NUM_SPECIES}_{START_RANK}_species"
    print(f"Species labels: {species_labels.keys()}")

    train, val, weights = datacreator.create_dataset(START_RANK)

    # Create datasets
    train_dataset = CustomDataset(train, train=True, img_size=IMG_SIZE)
    val_dataset = CustomDataset(val, train=False, img_size=IMG_SIZE)
    torch.save(val_dataset, "val_dataset.pt")

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

    # convert weights from dict to tensor
    if weights is not None:
        weights_tensor = torch.tensor(
            [weights.get(i, 1.0) for i in range(NUM_SPECIES)],
            dtype=torch.float32,
        )
    else:
        weights_tensor = None
    print(f"weights tensor: {weights_tensor}")

    # Build model - IMPORTANT: Use TOTAL_CLASSES instead of NUM_SPECIES
    model, image_size, description = build_model(net_id=MODEL_NAME, pretrained=True)
    in_features = model.classifier.linear.in_features  # type : ignore
    model.classifier.linear = torch.nn.Linear(  # type: ignore
        in_features, NUM_SPECIES
    )  # Changed this line

    print(f"Model configured for {NUM_SPECIES} classes")
    # Calculating weight for criterion for imbalanced dataset
    class_weights = torch.tensor(
        [1.0] * NUM_SPECIES, dtype=torch.float32
    )  # Placeholder for class weights
    # Set up training

    device = get_device()
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    model.to(device)

    # Training loop
    best_acc = -1.0
    best_f1 = -1.0
    for epoch in range(NUM_EPOCHS):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, macro_f1 = train_validate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        print(
            f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val acc: {val_acc:.4f} Val F1: {macro_f1:.4f}"
        )
        if macro_f1 > best_f1 or (macro_f1 == best_f1 and val_acc > best_acc):
            start_save = time.perf_counter()
            best_acc = val_acc
            best_f1 = macro_f1
            save_model(model, f"{NAME}", OUTPUT_DIR, device, IMG_SIZE)
            end_save = time.perf_counter()
            print(f"Save time: {end_save - start_save:.2f}s")
        end = time.perf_counter()
        print(f"Total time: {end - start:.2f}s")
    print(f"Best accuracy: {best_acc} with F1-score: {best_f1}")
