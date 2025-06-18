import os
import torch
import onnx
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from utilities import manifest_generator_wrapper
import logging
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_pytorch_model(model_path, save_path, calibration_loader):
    """Quantize PyTorch model to INT4"""
    print(f"Loading PyTorch model from {model_path}")
    # Set weights_only=False as we're loading our own trusted models
    model = torch.load(model_path, weights_only=False)
    model.eval()
    
    # Setup for dynamic quantization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare for quantization
    print("Applying quantization (closest to INT4 supported by PyTorch)...")
    
    # Use dynamic quantization (closest to INT4 available in standard PyTorch)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8  # Lowest supported by standard PyTorch
    )
    
    # Save quantized model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(quantized_model, save_path)
    print(f"Saved quantized PyTorch model to {save_path}")
    
    return quantized_model

def optimize_onnx_model(model_path, output_path):
    """Basic optimization for ONNX model before quantization"""
    print("Optimizing ONNX model...")
    
    try:
        # Load the model
        model = onnx.load(model_path)
        
        # Use ONNX's own optimizer
        from onnx import optimizer
        
        # List of optimization passes to apply
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'fuse_consecutive_transposes',
            'fuse_bn_into_conv'
        ]
        
        # Apply optimizations
        optimized_model = optimizer.optimize(model, passes)
        
        # Save the optimized model
        onnx.save(optimized_model, output_path)
        
        print(f"Saved optimized model to {output_path}")
        return output_path
    except Exception as e:
        print(f"Model optimization failed: {e}")
        print("Continuing with original model")
        return model_path

def quantize_onnx_model(model_path, save_path, calibration_loader):
    """Quantize ONNX model to INT4 (or closest supported precision)"""
    print(f"Loading ONNX model from {model_path}")
    
    # Create directory for quantized model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Optimize the model first with our custom function
    preprocessed_path = os.path.join(
        os.path.dirname(save_path),
        f"{os.path.splitext(os.path.basename(model_path))[0]}_optimized.onnx"
    )
    processed_model_path = optimize_onnx_model(model_path, preprocessed_path)
    
    print("Applying INT8 quantization (lowest precision supported by standard ONNX tools)...")
    
    try:
        # Use dynamic quantization with the optimized model
        quantize_dynamic(
            model_input=processed_model_path,
            model_output=save_path,
            weight_type=QuantType.QInt8,
            # Specify ops to quantize if available in this version
            op_types_to_quantize=['Conv', 'MatMul', 'Gemm', 'Add']
        )
        print(f"Saved quantized ONNX model to {save_path}")
        
        # Clean up the intermediate optimized model
        if processed_model_path != model_path and os.path.exists(processed_model_path):
            os.remove(processed_model_path)
            print(f"Removed intermediate optimized model: {processed_model_path}")
            
    except Exception as e:
        print(f"Quantization failed: {e}")
        
        # Try again with fewer parameters if the first attempt failed
        try:
            print("Trying simplified quantization...")
            quantize_dynamic(
                model_input=model_path,  # Use original model directly
                model_output=save_path,
                weight_type=QuantType.QInt8
            )
            print(f"Saved quantized ONNX model to {save_path}")
        except Exception as e2:
            print(f"Simplified quantization failed: {e2}")
            print("Copying original model as quantization failed")
            import shutil
            shutil.copy(model_path, save_path)
            print(f"Copied original model to {save_path}")

def prepare_calibration_data():
    """Prepare calibration dataset for quantization"""
    print("Preparing calibration data...")
    _, _, val, _, _ = manifest_generator_wrapper(0.3, export=False)
    
    # Create a smaller calibration dataset
    calibration_dataset = CustomDataset(val[:50], train=False, img_size=(160, 160))
    
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
    )
    
    return calibration_loader

def quantize_all_models(models_dir="models", output_dir="models/quantized"):
    """Quantize all models in the models directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare calibration data for quantization
    calibration_loader = prepare_calibration_data()
    
    # Find all model files
    pytorch_models = [f for f in os.listdir(models_dir) if f.endswith('.pth') and os.path.isfile(os.path.join(models_dir, f))]
    onnx_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx') and os.path.isfile(os.path.join(models_dir, f))]
    
    # Quantize PyTorch models
    print(f"Found {len(pytorch_models)} PyTorch models")
    for model_file in pytorch_models:
        model_path = os.path.join(models_dir, model_file)
        save_path = os.path.join(output_dir, f"{os.path.splitext(model_file)[0]}_int4.pth")
        
        print(f"\nQuantizing PyTorch model: {model_file}")
        try:
            quantize_pytorch_model(model_path, save_path, calibration_loader)
        except Exception as e:
            print(f"Error quantizing PyTorch model {model_file}: {e}")
    
    # Quantize ONNX models
    print(f"\nFound {len(onnx_models)} ONNX models")
    for model_file in onnx_models:
        model_path = os.path.join(models_dir, model_file)
        save_path = os.path.join(output_dir, f"{os.path.splitext(model_file)[0]}_int4.onnx")
        
        print(f"\nQuantizing ONNX model: {model_file}")
        try:
            quantize_onnx_model(model_path, save_path, calibration_loader)
        except Exception as e:
            print(f"Error quantizing ONNX model {model_file}: {e}")

if __name__ == "__main__":
    print("Starting model quantization to INT4 (or closest supported precision)...")
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    quantize_all_models()
    print("\nQuantization complete!")