import json
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import timeit
import argparse
import os

import onnxruntime as ort
import scipy.special
from onnxruntime import InferenceSession
from tqdm import tqdm

from pipeline.utility import (
    manifest_generator_wrapper,
    preprocess_eval_opencv,
)


class InferenceBenchmarkSingleModel:
    def __init__(
        self,
        model_path: str,
        global_data_manifests: List[Tuple[str, int]],
        global_species_labels: Dict[int, str],
        model_species_labels: Dict[int, str],
        model_input_size: Tuple[int, int] = (224, 224),
        is_big_inception_v3: bool = False,
        providers: str = "CPUExecutionProvider"
    ) -> None:
        self.session = ort.InferenceSession(model_path, providers=[providers])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = model_input_size

        self.global_data_manifests = global_data_manifests
        self.global_species_labels = global_species_labels
        self.global_labels_images: Dict[int, List[str]] = defaultdict(list)
        for image_path, species_id in self.global_data_manifests:
            self.global_labels_images[species_id].append(image_path)
        self.global_total_images = sum(len(imgs) for imgs in self.global_labels_images.values())
        self.global_species_probs = {
            int(species_id): len(images) / self.global_total_images
            for species_id, images in self.global_labels_images.items()
        }

        self.is_big_incv3 = is_big_inception_v3
        self.model_species_labels = model_species_labels
        self.other_labels = self._get_other_label()
    
    
    def _get_other_label(self) -> int:
        species_labels_flip: Dict[str, int] = dict((v, k) for k, v in self.model_species_labels.items())
        return species_labels_flip.get("Other", -1)


    def _create_stratified_weighted_sample(self, sample_size: int):
        sampled_species = list(self.global_species_labels.keys())
        remaining_k: int = sample_size - len(sampled_species)
        sampled_species += random.choices(
            population=sampled_species,
            weights=[self.global_species_probs[int(sid)] for sid in self.global_species_labels.keys()],
            k=remaining_k
        )
        random.shuffle(sampled_species)
        return [int(label) for label in sampled_species] 


    def _infer_one(self, image_path: str) -> Optional[Tuple[int, float, float]]:
        session: InferenceSession = self.session
        input_size = self.input_size
        input_name = self.input_name
        if self.is_big_incv3:
            is_incv3 = True
        else:
            is_incv3 = False

        try:
            img = preprocess_eval_opencv(image_path, *input_size, is_inception_v3=is_incv3)
            start = timeit.default_timer()
            outputs = session.run(None, {input_name: img})
            end = timeit.default_timer() - start
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob, end
        except Exception as e:
            print(e)
            return None


    def infer(self, image_path: str, ground_truth: int):
        result = self._infer_one(image_path)
        if result is None:
            print(f"Model returns no result for {image_path}")
            return None

        return result[0], result[2]


    def run(
        self,
        num_runs: int,
        sample_size: int,
        output_file: str = "./pred_other_result.json"
    ):
        all_elapsed_times: List[float] = []
        other_data: List[str] = []
        
        print(f"Starting benchmark with {num_runs} runs, {sample_size} samples per run")
        print(f"Input size: {self.input_size}")
        print(f"Provider: {self.session.get_providers()}")
        
        for run in range(num_runs):
            elapsed_times : List[float] = []
            sampled_species = self._create_stratified_weighted_sample(sample_size)

            for species_id in tqdm(sampled_species, desc=f"Run {run + 1}/{num_runs}", leave=False):
                image_list = self.global_labels_images[int(species_id)]
                if not image_list:
                    print("No image found")
                    continue
                image_path = random.choice(image_list)
                result = self.infer(image_path, species_id)
                if result is not None:
                    pred, elapsed = result
                    elapsed_times.append(elapsed)
                    if pred == self.other_labels:
                        other_data.append(image_path)

            all_elapsed_times.extend(elapsed_times)
            print(f"Run {run + 1}/{num_runs} completed: {len(elapsed_times)} samples processed")

        total_time_ms = sum(all_elapsed_times) * 1000
        avg_time_ms = np.mean(all_elapsed_times) * 1000
        std_time_ms = np.std(all_elapsed_times) * 1000
        fps = 1.0 / np.mean(all_elapsed_times)
        
        print(f"\n{'='*50}")
        print("INFERENCE PERFORMANCE RESULTS")
        print(f"{'='*50}")
        print(f"Model: {os.path.basename(self.session._model_path) if hasattr(self.session, '_model_path') else 'Unknown'}")
        print(f"Input size: {self.input_size}")
        print(f"Provider: {self.session.get_providers()[0]}")
        print(f"Total samples: {len(all_elapsed_times)}")
        print(f"Total inference time: {total_time_ms:.2f} ms")
        print(f"Average time per image: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms")
        print(f"Throughput (FPS): {fps:.2f} images/sec")
        print(f"Images predicted as 'Other': {len(other_data)}")
        print(f"{'='*50}")
        
        # Save results
        with open(output_file, "w") as output:
            json.dump(other_data, output, indent=2)
        print(f"Results saved to: {output_file}")
        
        return {
            'total_samples': len(all_elapsed_times),
            'total_time_ms': total_time_ms,
            'avg_time_ms': avg_time_ms,
            'std_time_ms': std_time_ms,
            'fps': fps,
            'other_predictions': len(other_data),
            'input_size': self.input_size
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX Model Inference Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="models/mcunet_haute_garonne_other_20_species_q8.onnx",
        help="Path to the ONNX model file"
    )
    
    # Input size configuration
    parser.add_argument(
        "--input-size", 
        type=int, 
        nargs=2, 
        default=[224, 224],
        metavar=('HEIGHT', 'WIDTH'),
        help="Input image size (height width). Common sizes: 160 160, 224 224, 299 299"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--num-runs", 
        type=int, 
        default=1,
        help="Number of benchmark runs to perform"
    )
    
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=1000,
        help="Number of samples per run"
    )
    
    # Model type
    parser.add_argument(
        "--inception-v3", 
        action="store_true",
        help="Use Inception v3 preprocessing"
    )
    
    # Execution provider
    parser.add_argument(
        "--provider", 
        type=str, 
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"],
        help="ONNX Runtime execution provider"
    )
    
    # Data configuration
    parser.add_argument(
        "--data-fraction", 
        type=float, 
        default=1.0,
        help="Fraction of dataset to use (0.0 to 1.0)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="./pred_other_result.json",
        help="Output file for 'Other' predictions"
    )
    
    # Verbose output
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments"""
    # Check model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Validate input size
    if len(args.input_size) != 2:
        raise ValueError("Input size must be exactly 2 values (height width)")
    
    if any(size <= 0 for size in args.input_size):
        raise ValueError("Input size values must be positive")
    
    # Validate data fraction
    if not 0.0 < args.data_fraction <= 1.0:
        raise ValueError("Data fraction must be between 0.0 and 1.0")
    
    # Validate runs and samples
    if args.num_runs <= 0:
        raise ValueError("Number of runs must be positive")
    
    if args.sample_size <= 0:
        raise ValueError("Sample size must be positive")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return True


def main():
    args = parse_args()
    
    try:
        validate_args(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    
    if args.verbose:
        print("Configuration:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print()
    
    # Load data
    print("Loading dataset manifests...")
    global_image_data, _, _, global_species_labels, global_species_composition = manifest_generator_wrapper(args.data_fraction)
    print(f"Loaded {len(global_image_data)} images from {len(global_species_labels)} species")
    
    # Initialize benchmark
    print(f"Initializing benchmark with model: {args.model_path}")
    pipeline = InferenceBenchmarkSingleModel(
        model_path=args.model_path,
        global_data_manifests=global_image_data,
        global_species_labels=global_species_labels,
        model_species_labels=global_species_labels,
        model_input_size=tuple(args.input_size),
        is_big_inception_v3=args.inception_v3,
        providers=args.provider
    )
    
    # Run benchmark
    results = pipeline.run(
        num_runs=args.num_runs,
        sample_size=args.sample_size,
        output_file=args.output_file
    )
    
    # Save benchmark summary
    summary_file = args.output_file.replace('.json', '_summary.json')
    summary = {
        'config': vars(args),
        'results': results,
        'timestamp': timeit.time.time()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Benchmark summary saved to: {summary_file}")
    return 0


if __name__ == "__main__":
    exit(main())
