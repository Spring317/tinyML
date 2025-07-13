import json
import os
from tqdm import tqdm
from typing import Tuple, Dict
import random
import yaml
# from get_num_class import save_class_counts_sorted_json


class DatasetCreator:
    def __init__(
        self,
        class_counter_path: str = "data_prep/class_counts_sorted.json",
        number_of_dominant_classes: int = 3,
    ):
        self.class_counter_path = class_counter_path
        self.number_of_dominant_classes = number_of_dominant_classes - 1
        # self.save_class_counts_sorted_json = save_class_counts_sorted_json

    def calculate_weight_samples(
        self, label_counts: Dict[int, int]
    ) -> Dict[int, float]:
        """Calculates the weight for each class based on the number of samples.

        Args:
            label_counts (Dict[int, int]): Dictionary with class labels as keys and sample counts as values.

        Returns:
            Dict[int, float]: Dictionary with class labels as keys and weights as values.
        """
        total_samples = sum(label_counts.values())
        weights = {
            label: total_samples / count for label, count in label_counts.items()
        }
        return weights

    def split_dataset(
        self, dataset: list[Dict[str, int]], train_ratio: float = 0.8
    ) -> Tuple[list, list]:
        """Splits the dataset into training and validation sets based on the specified ratio.

        Args:
            train_ratio (float): Ratio of training data to total data (default: 0.8)

        Returns:
            Tuple containing training dataset and validation dataset
        """
        # with open(self.class_counter_path, 'r', encoding='utf-8') as f:
        #     class_counts = json.load(f)

        # dataset = self.create_dataset()

        # Shuffle dataset
        random.shuffle(dataset)

        split_index = int(len(dataset) * train_ratio)
        train_dataset = dataset[:split_index]
        val_dataset = dataset[split_index:]

        return train_dataset, val_dataset

    def create_dataset(
        self, start_rank: int = 0, full_dataset: bool = False
    ) -> Tuple[list, list, Dict[int, float]]:
        """
        Creates a dataset by loading the class counts from a JSON file.

        Strategy:
        - Take n consecutive dominant species starting from start_rank as classes 0, 1, 2, ..., n-1
        - Put ALL OTHER species into class n as "Other"

        Args:
            start_rank: Starting rank (0-indexed) for selecting the n dominant species
            full_dataset: If True, uses all species; if False, uses only the dominant species.
        Returns:
            List containing dataset with image paths and labels.
        """
        if not os.path.exists(self.class_counter_path):
            raise FileNotFoundError(
                f"Class counts file not found: {self.class_counter_path}"
            )

        with open(self.class_counter_path, "r", encoding="utf-8") as f:
            class_counts = json.load(f)

        dataset = []

        label_counts = {}
        # Load data_path from config.yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        data_path = config["paths"]["src_dataset"]
        print(f"Using data path: {data_path}")
        # Get species lists
        all_species = class_counts["class_counts_sorted"]
        if not full_dataset:
            # Selected n species starting from start_rank -> will become classes 0, 1, 2, ..., n-1
            end_rank = start_rank + self.number_of_dominant_classes
            selected_species = all_species[start_rank:end_rank]

            # Everything else -> will become "Other" class (class n)
            other_species = all_species[:start_rank] + all_species[end_rank:]

            # Other class label is number_of_dominant_classes
            other_class_label = self.number_of_dominant_classes

            print("🏷️  Dataset Creation Strategy:")
            print("=" * 70)
            print(f"Number of dominant classes: {self.number_of_dominant_classes}")
            print(
                f"Selected range: Rank {start_rank + 1}-{end_rank} (indices {start_rank}-{end_rank - 1})"
            )
            print("=" * 70)
            print(f"MAIN CLASSES (0-{self.number_of_dominant_classes - 1}):")
            for i, species in enumerate(selected_species):
                rank = start_rank + i + 1
                print(
                    f"  Class {i}: {species['class_name']} (Rank {rank}, {species['sample_count']} samples)"
                )

            print(f"\nOTHER CLASS ({other_class_label}):")
            total_other_samples = sum(
                species["sample_count"] for species in other_species
            )
            print(f"  All other {len(other_species)} species")
            print(f"  Total samples: {total_other_samples}")
            print("=" * 70)

            # Process selected n species as main classes (0, 1, 2, ..., n-1)
            for class_idx, species in enumerate(
                tqdm(selected_species, desc="Processing main classes")
            ):
                class_name = species["class_name"]
                class_path = os.path.join(data_path, f"{class_name}")
                print(f"Using class path: {class_path}")
                if os.path.exists(class_path):
                    for sample in os.listdir(class_path):
                        if sample.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            dataset.append(
                                {
                                    "image": os.path.join(class_path, sample),
                                    "label": class_idx,
                                }
                            )
                else:
                    print(f"⚠️  Warning: Path not found for {class_name}")

            # Process all other species as "Other" class (class number_of_dominant_classes)
            for species in tqdm(
                other_species, desc=f"Processing 'Other' class ({other_class_label})"
            ):
                class_name = species["class_name"]
                class_path = os.path.join(data_path, class_name)

                if os.path.exists(class_path):
                    for sample in os.listdir(class_path):
                        if sample.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            dataset.append(
                                {
                                    "image": os.path.join(class_path, sample),
                                    "label": other_class_label,  # "Other" class
                                }
                            )
                else:
                    print(f"⚠️  Warning: Path not found for {class_name}")

            # Print dataset summary
            for item in dataset:
                label = item["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            print("\nDATASET SUMMARY:")
            print("=" * 50)
            for label in sorted(label_counts.keys()):
                if label < other_class_label:
                    class_name = selected_species[label]["class_name"]
                    rank = start_rank + label + 1
                    print(
                        f"Class {label} ({class_name}, Rank {rank}): {label_counts[label]} samples"
                    )
                else:
                    print(
                        f"Class {label} (Other - all remaining): {label_counts[label]} samples"
                    )

            print(f"Total dataset size: {len(dataset)} samples")
            print(
                f"Total classes: {other_class_label + 1} ({self.number_of_dominant_classes} main + 1 other)"
            )
        else:
            # Use all species as main classes (0, 1, 2, ..., n-1)
            print("Using full dataset with all species as main classes")
            for class_idx, species in enumerate(
                tqdm(all_species, desc="Processing full dataset")
            ):
                class_name = species["class_name"]
                class_path = os.path.join(data_path, f"{class_name}")
                print(f"Using class path: {class_path}")
                if os.path.exists(class_path):
                    for sample in os.listdir(class_path):
                        if sample.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            dataset.append(
                                {
                                    "image": os.path.join(class_path, sample),
                                    "label": class_idx,
                                }
                            )
                else:
                    print(f"⚠️  Warning: Path not found for {class_name}")

            # Print dataset summary
            for item in dataset:
                label = item["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            print("\nDATASET SUMMARY:")
            print("=" * 50)
            for label in sorted(label_counts.keys()):
                class_name = all_species[label]["class_name"]
                rank = label + 1
                print(
                    f"Class {label} ({class_name}, Rank {rank}): {label_counts[label]} samples"
                )

            print(f"Total dataset size: {len(dataset)} samples")
            print(f"Total classes: {len(all_species)}")

        train, val = self.split_dataset(dataset)
        print(f"Training set size: {len(train)} samples")
        print(f"Validation set size: {len(val)} samples")

        weights = self.calculate_weight_samples(label_counts)
        return train, val, weights
