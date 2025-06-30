import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import time
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark TinyML quantized model with single image')
    parser.add_argument('--input_size', type=int, choices=[28, 32, 64, 128, 160, 224], 
                       default=160, help='Input image size (default: 160)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results (default: ./results)')
    parser.add_argument('--save_format', type=str, choices=['png', 'pdf', 'svg'],
                       default='png', help='Format to save plots (default: png)')
    parser.add_argument('--model_path', type=str, 
                       default='models/mcunet_haute_garonne_2_species_q8.onnx',
                       help='Path to quantized ONNX model')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Specific image path for inference (REQUIRED)')
    parser.add_argument('--iterations', type=int, default=5000,
                       help='Number of inference iterations (default: 5000)')
    return parser.parse_args()

def get_process_info():
    """Get current process information"""
    pid = os.getpid()
    return pid

def print_process_info():
    """Print current process information"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        info = {
            'pid': os.getpid(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
        }
        print(f"Process ID: {info['pid']}")
        print(f"Memory Usage: {info['memory_mb']:.2f} MB")
        print(f"CPU Usage: {info['cpu_percent']:.2f}%")
        print(f"Number of Threads: {info['num_threads']}")
    except ImportError:
        pid = get_process_info()
        print(f"Process ID: {pid}")

def get_train_transforms(img_size=(160, 160)):
    """
    Get the same transforms used during training to ensure consistency
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  # Converts PIL to tensor and normalizes to [0,1]
    ])

def load_single_image(image_path, image_size=(160, 160)):
    """
    Load a single specified image for inference testing
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    transform = get_train_transforms(image_size)
    
    try:
        print(f"Loading image: {image_path}")
        
        # Load image using PIL (same as training)
        img = Image.open(image_path).convert("RGB")
        print(f"Original image size: {img.size}")
        
        # Apply the same transforms as training
        img_tensor = transform(img)
        
        # Convert to numpy for ONNX Runtime
        img_np = img_tensor.numpy()
        
        # Add batch dimension
        img_np = np.expand_dims(img_np, 0)
        
        print(f"Processed image shape: {img_np.shape}")
        print(f"Image value range: [{img_np.min():.3f}, {img_np.max():.3f}]")
        
        return img_np, image_path
        
    except Exception as e:
        raise RuntimeError(f"Error processing image {image_path}: {e}")

def benchmark_quantized_model(model_path, input_data, image_path, num_iterations=5000):
    """
    Perform inference benchmark with quantized model on single image
    """
    print(f"\n=== Quantized Model Inference Benchmark ===")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Iterations: {num_iterations}")
    print(f"Input shape: {input_data.shape}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Get model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Create ONNX Runtime session
    try:
        # Use CPU provider for consistent benchmarking
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"✓ ONNX Runtime session created successfully")
        print(f"✓ Using providers: {session.get_providers()}")
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
        
        # Print model input/output shapes
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        print(f"Expected input shape: {input_shape}")
        print(f"Expected output shape: {output_shape}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to create ONNX Runtime session: {e}")
    
    # Warm-up runs (important for accurate benchmarking)
    print("\nPerforming warm-up runs...")
    warmup_times = []
    for i in range(50):  # More warm-up runs for stable performance
        start = time.perf_counter()
        _ = session.run([output_name], {input_name: input_data})
        end = time.perf_counter()
        warmup_times.append((end - start) * 1000)
    
    print(f"✓ Warm-up completed (avg: {np.mean(warmup_times):.3f}ms)")
    
    # Main benchmark runs
    print(f"\nRunning {num_iterations} inference iterations...")
    inference_times = []
    predictions = []
    
    # Use high precision timer
    start_total = time.perf_counter()
    
    for i in range(num_iterations):
        start_time = time.perf_counter()
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_data})
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
        
        # Store first prediction for consistency check
        if i == 0:
            predictions.append(outputs[0])
        
        # Print progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            avg_time = np.mean(inference_times[-1000:])
            print(f"  Progress: {i + 1:4d}/{num_iterations} - Avg time (last 1000): {avg_time:.3f}ms")
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    
    print(f"\n=== Benchmark Results ===")
    print(f"Total iterations: {num_iterations}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average inference time: {np.mean(inference_times):.4f} ms")
    print(f"Median inference time: {np.median(inference_times):.4f} ms")
    print(f"Min inference time: {np.min(inference_times):.4f} ms")
    print(f"Max inference time: {np.max(inference_times):.4f} ms")
    print(f"Std deviation: {np.std(inference_times):.4f} ms")
    print(f"95th percentile: {np.percentile(inference_times, 95):.4f} ms")
    print(f"99th percentile: {np.percentile(inference_times, 99):.4f} ms")
    print(f"Throughput: {num_iterations / total_time:.2f} FPS")
    
    # Show prediction for consistency check
    if predictions:
        pred = predictions[0][0]
        predicted_class = np.argmax(pred)
        confidence = np.max(pred)
        print(f"\n=== Prediction Results ===")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Top 3 predictions:")
        top3_indices = np.argsort(pred)[-3:][::-1]
        for idx in top3_indices:
            print(f"  Class {idx}: {pred[idx]:.4f}")
    
    return {
        'avg_time_ms': np.mean(inference_times),
        'median_time_ms': np.median(inference_times),
        'min_time_ms': np.min(inference_times),
        'max_time_ms': np.max(inference_times),
        'std_time_ms': np.std(inference_times),
        'p95_time_ms': np.percentile(inference_times, 95),
        'p99_time_ms': np.percentile(inference_times, 99),
        'throughput_fps': num_iterations / total_time,
        'total_time_s': total_time,
        'all_times': inference_times,
        'prediction': predictions[0] if predictions else None
    }

def plot_inference_times(inference_times, num_iterations, output_dir='./results'):
    """Plot inference time distribution and timeline"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Histogram of inference times
    axes[0, 0].hist(inference_times, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Inference Time (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Inference Time Distribution ({num_iterations} iterations)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics to histogram
    mean_time = np.mean(inference_times)
    median_time = np.median(inference_times)
    axes[0, 0].axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.3f}ms')
    axes[0, 0].axvline(median_time, color='green', linestyle='--', label=f'Median: {median_time:.3f}ms')
    axes[0, 0].legend()
    
    # Plot 2: Timeline of inference times
    axes[0, 1].plot(inference_times, alpha=0.7, color='blue', linewidth=0.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Inference Time (ms)')
    axes[0, 1].set_title('Inference Time Timeline')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    axes[1, 0].boxplot(inference_times, vert=True)
    axes[1, 0].set_ylabel('Inference Time (ms)')
    axes[1, 0].set_title('Inference Time Distribution (Box Plot)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Running average
    window_size = max(50, num_iterations // 100)
    running_avg = np.convolve(inference_times, np.ones(window_size)/window_size, mode='valid')
    axes[1, 1].plot(range(window_size-1, len(inference_times)), running_avg, 
                    color='red', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Running Average (ms)')
    axes[1, 1].set_title(f'Running Average (window={window_size})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'inference_benchmark_detailed.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Detailed benchmark plots saved to: {plot_path}")

def save_benchmark_results(results, args, image_path, output_dir):
    """Save benchmark results to file"""
    results_path = os.path.join(output_dir, f'benchmark_results_{args.iterations}_iterations.txt')
    
    with open(results_path, 'w') as f:
        f.write(f"TinyML Quantized Model Benchmark Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Input size: {args.input_size}x{args.input_size}\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Process ID: {get_process_info()}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nPerformance Metrics:\n")
        f.write(f"  Average time: {results['avg_time_ms']:.4f} ms\n")
        f.write(f"  Median time: {results['median_time_ms']:.4f} ms\n")
        f.write(f"  Min time: {results['min_time_ms']:.4f} ms\n")
        f.write(f"  Max time: {results['max_time_ms']:.4f} ms\n")
        f.write(f"  Std deviation: {results['std_time_ms']:.4f} ms\n")
        f.write(f"  95th percentile: {results['p95_time_ms']:.4f} ms\n")
        f.write(f"  99th percentile: {results['p99_time_ms']:.4f} ms\n")
        f.write(f"  Throughput: {results['throughput_fps']:.2f} FPS\n")
        f.write(f"  Total time: {results['total_time_s']:.3f} seconds\n")
        
        if results['prediction'] is not None:
            pred = results['prediction'][0]
            predicted_class = np.argmax(pred)
            confidence = np.max(pred)
            f.write(f"\nPrediction Results:\n")
            f.write(f"  Predicted class: {predicted_class}\n")
            f.write(f"  Confidence: {confidence:.4f}\n")
            
            top3_indices = np.argsort(pred)[-3:][::-1]
            f.write(f"  Top 3 predictions:\n")
            for idx in top3_indices:
                f.write(f"    Class {idx}: {pred[idx]:.4f}\n")
    
    print(f"✓ Benchmark results saved to: {results_path}")

def main():
    args = parse_arguments()
    
    # Print process information at start
    print("=== Process Information ===")
    print_process_info()
    print("=" * 30)
    
    print(f"=== TinyML Single Image Benchmark ===")
    print(f"Input size: {args.input_size}x{args.input_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Image: {os.path.basename(args.image_path)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load single specified image
    print(f"\n1. Loading single image...")
    try:
        test_image, image_path = load_single_image(
            args.image_path, (args.input_size, args.input_size)
        )
        print(f"✓ Successfully loaded image")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return
    
    # Run benchmark
    print(f"\n2. Running {args.iterations} inference iterations...")
    try:
        results = benchmark_quantized_model(
            args.model_path, 
            test_image, 
            image_path,
            args.iterations
        )
        
        # Create plots
        plot_inference_times(results['all_times'], args.iterations, args.output_dir)
        
        # Save results
        save_benchmark_results(results, args, image_path, args.output_dir)
        
        print(f"\n✓ Benchmark completed successfully!")
        print(f"✓ All results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
