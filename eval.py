import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_bin import BinaryInsectDataset
from CustomDataset import CustomDataset
from utilities import get_device

def evaluate_model(model_path, val_dataset_path, batch_size=32, num_workers=4, output_dir="evaluation_results"):
    """
    Comprehensive evaluation of a trained model with class-specific metrics.
    
    Args:
        model_path (str): Path to the saved PyTorch model
        val_dataset_path (str): Path to the validation dataset
        batch_size (int): Batch size for evaluation
        num_workers (int): Number of workers for data loading
        output_dir (str): Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and dataset
    device = get_device()
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()
    
    val_dataset = torch.load(val_dataset_path, weights_only=False, map_location=device)
    
    # Get class names
    # Try different approaches to get class names
    try:
        # First attempt: check if the dataset has a class_to_idx attribute
        class_to_idx = getattr(val_dataset, 'class_to_idx', None)
        if class_to_idx:
            class_names = list(class_to_idx.keys())
        else:
            # Second attempt: check for species_labels attribute
            species_labels = getattr(val_dataset, 'species_labels', None)
            if species_labels:
                class_names = list(species_labels.keys())
            else:
                # Third attempt: check for classes attribute
                classes = getattr(val_dataset, 'classes', None)
                if classes:
                    class_names = classes
                else:
                    # Fallback: use numeric class labels
                    all_labels_set = set(val_dataset.tensors[1].numpy() if hasattr(val_dataset, 'tensors') else 
                                        [label for _, label in val_dataset])
                    class_names = [f"Class {i}" for i in sorted(all_labels_set)]
    except Exception as e:
        print(f"Warning: Could not extract class names automatically: {e}")
        print("Using numeric class indices instead.")
        # Extract unique labels from the dataset
        all_labels = []
        for _, label in val_dataset:
            all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        unique_labels = sorted(set(all_labels))
        class_names = [f"Class {i}" for i in unique_labels]
    
    num_classes = len(class_names)
    
    print(f"Loaded model from {model_path}")
    print(f"Loaded validation dataset with {len(val_dataset)} samples and {num_classes} classes")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    all_probs = []  # For confidence scores
    
    # Evaluate model
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            # Get predicted class
            _, preds = torch.max(outputs, 1)
            
            # Convert to probabilities with softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision = precision_score(all_labels, all_preds, average='macro')
    overall_recall = recall_score(all_labels, all_preds, average='macro')
    overall_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n===== OVERALL METRICS =====")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision (macro): {overall_precision:.4f}")
    print(f"Recall (macro): {overall_recall:.4f}")
    print(f"F1 Score (macro): {overall_f1:.4f}")
    
    # Get detailed classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        labels=range(num_classes),
        target_names=class_names,
        output_dict=True
    )
    
    # Convert classification report to DataFrame for easier handling
    report_df = pd.DataFrame(report).transpose()
    
    # Save classification report
    report_df.to_csv(f"{output_dir}/classification_report.csv")
    
    # Print class-specific metrics
    print("\n===== CLASS-SPECIFIC METRICS =====")
    
    # Create a DataFrame for class metrics
    class_metrics = []
    for i in range(num_classes):
        class_name = class_names[i]
        # Handle different key formats in the classification report
        key = str(i)
        
        # Check if the key exists in the report, otherwise try class_name
        if key not in report:
            key = class_name
            # If that doesn't work either, skip this class
            if key not in report:
                print(f"Warning: Could not find metrics for class {i} ({class_name}) in the report")
                print(f"Available keys in report: {list(report.keys())}")
                continue
        
        metrics = {
            'class_name': class_name,
            'precision': report[key]['precision'],
            'recall': report[key]['recall'],
            'f1_score': report[key]['f1-score'],
            'support': report[key]['support'],
            'accuracy': accuracy_score(all_labels == i, all_preds == i),
        }
        class_metrics.append(metrics)
        
        print(f"Class: {class_name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    # Create DataFrame and save to CSV
    class_df = pd.DataFrame(class_metrics)
    class_df.to_csv(f"{output_dir}/class_metrics.csv", index=False)
    
    # Calculate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")
    
    # Calculate per-class confidence scores
    class_confidence = []
    for i in range(num_classes):
        # Get samples where the model predicted class i
        pred_class_i = all_preds == i
        if np.sum(pred_class_i) > 0:
            # Get the confidence scores for those predictions
            conf_scores = all_probs[pred_class_i, i]
            class_confidence.append({
                'class_name': class_names[i],
                'mean_confidence': np.mean(conf_scores),
                'min_confidence': np.min(conf_scores),
                'max_confidence': np.max(conf_scores),
                'std_confidence': np.std(conf_scores)
            })
    
    # Save confidence scores
    confidence_df = pd.DataFrame(class_confidence)
    confidence_df.to_csv(f"{output_dir}/class_confidence.csv", index=False)
    
    # Generate plots
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    
    # 2. Class F1 Scores
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x='class_name', y='f1_score', data=class_df)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_f1_scores.png", dpi=300)
    
    # 3. Precision & Recall by Class
    plt.figure(figsize=(14, 6))
    class_df_melted = pd.melt(class_df, id_vars=['class_name'], value_vars=['precision', 'recall'], var_name='metric', value_name='score')
    sns.barplot(x='class_name', y='score', hue='metric', data=class_df_melted)
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision and Recall by Class')
    plt.xticks(rotation=90)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall.png", dpi=300)
    
    # 4. Class Confidence Distribution
    if len(class_confidence) > 0:
        plt.figure(figsize=(14, 6))
        ax = sns.barplot(x='class_name', y='mean_confidence', data=confidence_df)
        plt.errorbar(
            x=np.arange(len(confidence_df)),
            y=confidence_df['mean_confidence'],
            yerr=confidence_df['std_confidence'],
            fmt='none',
            capsize=5,
            color='black'
        )
        plt.xlabel('Class')
        plt.ylabel('Mean Confidence Score')
        plt.title('Mean Prediction Confidence by Class')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_confidence.png", dpi=300)
    
    print(f"\nEvaluation complete. Results saved to {output_dir}/")
    
    return {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'class_metrics': class_df,
        'confusion_matrix': cm_df
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model with detailed metrics')
    parser.add_argument('--model', type=str, default="models/mcunet_haute_garonne_8_species.pth", help='Path to model file (.pth)')
    parser.add_argument('--dataset', type=str, default="val_dataset.pt", help='Path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--output', type=str, default="evaluation_results", help='Output directory for results')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        val_dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output
    )
