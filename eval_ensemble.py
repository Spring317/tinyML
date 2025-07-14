from pipe_model import PipeModel
from utility.scores import (
    get_classification_report,
    calculate_scores,
    calculate_other_precision_recall_f1,
)
from utility.utilities import get_device
from data_prep.data_loader import DataLoaderCreator
from data_prep.class_handler import get_class_info_for_evaluation
from models.model_handler import ModelHandler
import argparse
import pandas as pd
import torch


def eval_ensemble(
    model1_path: str,
    model2_path: str,
    dominant_threshold: float = 0.2,
    start_rank: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Evaluate the ensemble model on the provided dataloader.

    Args:
        model (PipeModel): The ensemble model to evaluate.
        dataloader (DataLoader): The dataloader containing the validation data.

    Returns:
        tuple: A tuple containing all labels and all predictions.
    """

    device = get_device()
    # Load the ensemble model
    model1 = torch.load(model1_path, map_location=device, weights_only=False)
    model2 = torch.load(model2_path, map_location=device, weights_only=False)

    model = PipeModel(model1, model2, 3, device=device)

    dataloader_creator = DataLoaderCreator(
        batch_size, num_workers, dominant_threshold, start_rank, full_dataset=True
    )

    # Get validation dataloader
    _, val_loader, _, _ = dataloader_creator.create_dataloader()

    # Initialize lists to store predictions and ground truth
    # Evaluate model
    model_handler = ModelHandler(device)

    # class_names, num_classes, class_to_idx = get_class_info_for_evaluation(
    #     start_rank=start_rank,
    #     number_of_dominant_classes=3,  # Default to 3 main classes + 1 "Other"
    #     model_path=model_path,
    # )

    print("Starting evaluation...")
    all_preds, all_labels, all_probs = model_handler.eval_one_epoch(
        model=model,
        dataloader=val_loader,
    )

    # Calculate overall metrics
    overall_accuracy, overall_precision, overall_recall, overall_f1 = calculate_scores(
        all_labels,  # type: ignore
        all_preds,  # type: ignore
    )

    print("\n===== OVERALL METRICS =====")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision (macro): {overall_precision:.4f}")
    print(f"Recall (macro): {overall_recall:.4f}")
    print(f"F1 Score (macro): {overall_f1:.4f}")

    # Get detailed classification report
    # report = get_classification_report(
    #     all_labels,  # type: ignore
    #     all_preds,  # type: ignore
    #     num_classes=num_classes,
    #     class_names=class_names,
    #     output_dict=True,
    # )
    #
    # # Convert classification report to DataFrame
    # report_df = pd.DataFrame(report).transpose()
    #
    # # Calculate precision and recall for the "Other" class if it exists
    # if num_classes > 3:  # Assuming last class is "Other"
    #     pred, rec, f1 = calculate_other_precision_recall_f1(
    #         all_labels, all_preds, num_classes
    #     )
    #     other_class_name = class_names[-1]  # Last class should be "Other"
    #     report_df.loc[other_class_name, "precision"] = pred
    #     report_df.loc[other_class_name, "recall"] = rec
    #     report_df.loc[other_class_name, "f1-score"] = f1
    #
    # # Save classification report
    # report_df.to_csv(f"{output_dir}/classification_report.csv")
    #
    # # Print class-specific metrics
    # print("\n===== CLASS-SPECIFIC METRICS =====")
    # class_metrics = []
    #
    # for i in range(num_classes):
    #     class_name = class_names[i]
    #     key = str(i)
    #
    #     # Check if the key exists in the report, otherwise try class_name
    #     if key not in report:
    #         key = class_name
    #         if key not in report:
    #             print(f"Warning: Could not find metrics for class {i} ({class_name})")
    #             continue
    #
    #     metrics = {
    #         "class_name": class_name,
    #         "precision": report[key]["precision"],
    #         "recall": report[key]["recall"],
    #         "f1_score": report[key]["f1-score"],
    #         "support": report[key]["support"],
    #         "accuracy": report[key].get("accuracy", 0.0),
    #     }
    #     class_metrics.append(metrics)
    #
    #     print(f"Class: {class_name}")
    #     print(f"  Precision: {metrics['precision']:.4f}")
    #     print(f"  Recall: {metrics['recall']:.4f}")
    #     print(f"  F1 Score: {metrics['f1_score']:.4f}")
    #     print(f"  Support: {metrics['support']}")
    #
    # # Create DataFrame and save to CSV
    # class_df = pd.DataFrame(class_metrics)
    # class_df.to_csv(f"{output_dir}/class_metrics.csv", index=False)
    #
    # # Calculate and save confusion matrix
    # cm = confusion_matrix(all_labels, all_preds)
    # cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    # cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")
    #
    # # Calculate per-class confidence scores
    # class_confidence = []
    # for i in range(num_classes):
    #     pred_class_i = all_preds == i
    #     if np.sum(pred_class_i) > 0:
    #         conf_scores = all_probs[pred_class_i, i]
    #         class_confidence.append(
    #             {
    #                 "class_name": class_names[i],
    #                 "mean_confidence": np.mean(conf_scores),
    #                 "min_confidence": np.min(conf_scores),
    #                 "max_confidence": np.max(conf_scores),
    #                 "std_confidence": np.std(conf_scores),
    #             }
    #         )
    #
    # # Save confidence scores
    # if class_confidence:
    #     confidence_df = pd.DataFrame(class_confidence)
    #     confidence_df.to_csv(f"{output_dir}/class_confidence.csv", index=False)
    #
    # # Generate plots
    # plot_confusion_matrix_heatmap(cm, class_names, output_dir)
    # plot_class_f1_scores(class_df, output_dir)
    # plot_class_precision_recall(class_df, output_dir)
    #
    # print(f"\nEvaluation complete. Results saved to {output_dir}/")
    #
    # return {
    #     "accuracy": overall_accuracy,
    #     "precision": overall_precision,
    #     "recall": overall_recall,
    #     "f1": overall_f1,
    #     "class_metrics": class_df,
    #     "confusion_matrix": cm_df,
    # }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model with detailed metrics"
    )
    parser.add_argument(
        "--model1",
        type=str,
        default="models/mcunet_haute_garonne_8_species.pth",
        help="Path to mcunetmodel file (.pth)",
    )

    parser.add_argument(
        "--model2",
        type=str,
        default="models/mcunet_haute_garonne_8_species.pth",
        help="Path to convnextmodel file (.pth)",
    )
    parser.add_argument(
        "--dominant_threshold",
        type=float,
        default=0.5,
        help="Threshold for dominant species classification",
    )
    parser.add_argument(
        "--start_rank", type=int, default=0, help="Starting rank for dataset creation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    eval_ensemble(
        model1_path=args.model1,
        model2_path=args.model2,
        dominant_threshold=args.dominant_threshold,
        start_rank=args.start_rank,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # output_dir=args.output,
    )
