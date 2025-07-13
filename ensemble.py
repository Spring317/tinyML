from pipe_model import PipeModel
import torch
from typing import List, Tuple
from data_prep.create_dataset import DatasetCreator
import pipe_model
from utilities import get_device, manifest_generator_wrapper
from CustomDataset import CustomDataset
from tqdm import tqdm


class EnsembleModel:
    """Perform evaluation of MCUnet and ConvNeXt models on a dataset."""

    def __init__(self, mcunet, num_dominnat_classes, convnext):
        """
        Initialize the EnsembleModel with two models.

        Args:
            mcunet: The first model (e.g., a MobileNetV2 or similar).
            num_dominnat_classes: Number of classes for the first model.
            convnext: The second model (e.g., ConvNeXt).
        """
        self.pipe_model = PipeModel(mcunet, num_dominnat_classes, convnext)
        self.device = get_device()

    def evaluate(self) -> Tuple[List[int], List[int]]:
        """Evaluate the ensemble model on the dataset."""

        _, _, _, species_labels, _ = manifest_generator_wrapper(1.0, export=True)

        # Load validation dataset

        num_classes = len(species_labels.keys())

        print(f"Number of species from manifest: {num_classes}")
        datacreator = DatasetCreator(number_of_dominant_classes=num_classes)
        # The actual number of classes in the dataset is NUM_SPECIES + 1 (including "Other" class)

        print(f"Species labels: {species_labels.keys()}")

        _, val, _ = datacreator.create_dataset(start_rank=0, full_dataset=True)
        val_dataset = CustomDataset(val, train=False, img_size=(160, 160))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        predictions = []
        labels = []

        self.pipe_model.eval()
        with torch.no_grad():
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Evaluating"):
                    images = images.to(self.device)
                    outputs = self.pipe_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.cpu().numpy())
                    labels.extend(labels.numpy())

        return predictions, labels
