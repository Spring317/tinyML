import os
import torch
import onnx
import numpy as np
from collections import Counter
import gc

def get_file_size_mb(file_path):
    """Get file size in MB"""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def verify_pytorch_quantization_safe(original_path, quantized_path):
    """Safely verify if PyTorch model is actually quantized"""
    print(f"\n----- PyTorch Model: {os.path.basename(quantized_path)} -----")
    
    # Check file sizes first
    orig_size = get_file_size_mb(original_path)
    quant_size = get_file_size_mb(quantized_path)
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Size reduction: {100 * (1 - quant_size/orig_size):.1f}%")
    
    try:
        # Load models one at a time and clear memory immediately
        print("Loading original model...")
        orig_model = torch.load(original_path, weights_only=False, map_location='cpu')
        
        # Get original model info
        orig_dtypes = Counter()
        orig_param_count = 0
        for name, param in orig_model.state_dict().items():
            orig_dtypes[str(param.dtype)] += 1
            orig_param_count += param.numel()
        
        # Clear original model from memory
        del orig_model
        gc.collect()
        
        print("Loading quantized model...")
        quant_model = torch.load(quantized_path, weights_only=False, map_location='cpu')
        
        # Get quantized model info
        quant_dtypes = Counter()
        quant_param_count = 0
        for name, param in quant_model.state_dict().items():
            quant_dtypes[str(param.dtype)] += 1
            quant_param_count += param.numel()
        
        # Clear quantized model from memory
        del quant_model
        gc.collect()
        
        print(f"\nParameter count - Original: {orig_param_count:,}, Quantized: {quant_param_count:,}")
        
        print("\nOriginal model parameter types:")
        for dtype, count in orig_dtypes.items():
            print(f"  {dtype}: {count} parameters")
        
        print("\nQuantized model parameter types:")
        for dtype, count in quant_dtypes.items():
            print(f"  {dtype}: {count} parameters")
        
        # Check if quantized model has int8/qint8 parameters
        is_quantized = any('int8' in dtype.lower() for dtype in quant_dtypes.keys())
        
        if is_quantized:
            print("\n✅ VERIFIED: Model contains quantized (INT8) parameters")
        else:
            print("\n❌ NOT VERIFIED: Model does not contain INT8 parameters")
            print("The model may use dynamic quantization (applied during inference only)")
            
    except Exception as e:
        print(f"Error loading PyTorch models: {e}")
        print("This might indicate memory issues or corrupted model files")

def verify_onnx_quantization_safe(original_path, quantized_path):
    """Safely verify if ONNX model is actually quantized"""
    print(f"\n----- ONNX Model: {os.path.basename(quantized_path)} -----")
    
    # Check file sizes
    orig_size = get_file_size_mb(original_path)
    quant_size = get_file_size_mb(quantized_path)
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Size reduction: {100 * (1 - quant_size/orig_size):.1f}%")
    
    try:
        # Load original model
        print("Analyzing original ONNX model...")
        orig_model = onnx.load(original_path)
        
        orig_ops = Counter(node.op_type for node in orig_model.graph.node)
        orig_init_types = Counter()
        
        for init in orig_model.graph.initializer:
            orig_init_types[str(init.data_type)] += 1
        
        # Clear from memory
        del orig_model
        gc.collect()
        
        # Load quantized model
        print("Analyzing quantized ONNX model...")
        quant_model = onnx.load(quantized_path)
        
        quant_ops = Counter(node.op_type for node in quant_model.graph.node)
        quant_init_types = Counter()
        
        for init in quant_model.graph.initializer:
            quant_init_types[str(init.data_type)] += 1
        
        # Clear from memory
        del quant_model
        gc.collect()
        
        print("\nTop 10 operation types in original model:")
        for op, count in orig_ops.most_common(10):
            print(f"  {op}: {count}")
        
        print("\nTop 10 operation types in quantized model:")
        for op, count in quant_ops.most_common(10):
            print(f"  {op}: {count}")
        
        # Check for quantization operations
        quant_nodes = ['QuantizeLinear', 'DequantizeLinear']
        has_quant_ops = any(op_name in quant_ops for op_name in quant_nodes)
        
        # Check for INT8 tensors (type 2 in ONNX)
        has_int8_tensors = '2' in quant_init_types
        
        print(f"\nQuantization operations found: {has_quant_ops}")
        print(f"INT8 tensors found: {has_int8_tensors}")
        
        if has_quant_ops or has_int8_tensors:
            print("\n✅ VERIFIED: Model contains quantization operations or INT8 tensors")
        else:
            print("\n❌ NOT VERIFIED: Model does not appear to be quantized")
            
    except Exception as e:
        print(f"Error analyzing ONNX model: {e}")

def simple_file_comparison(models_dir="models", quantized_dir="models/quantized"):
    """Simple file size comparison without loading models"""
    print("===== SIMPLE FILE SIZE COMPARISON =====")
    
    # Find all models
    all_files = os.listdir(models_dir)
    pytorch_models = [f for f in all_files if f.endswith('.pth')]
    onnx_models = [f for f in all_files if f.endswith('.onnx')]
    
    print("\nPyTorch Models:")
    for model_file in pytorch_models:
        base_name = os.path.splitext(model_file)[0]
        orig_path = os.path.join(models_dir, model_file)
        quant_path = os.path.join(quantized_dir, f"{base_name}_int4.pth")
        
        if os.path.exists(quant_path):
            orig_size = get_file_size_mb(orig_path)
            quant_size = get_file_size_mb(quant_path)
            reduction = 100 * (1 - quant_size/orig_size)
            print(f"  {model_file}: {orig_size:.2f} MB → {quant_size:.2f} MB ({reduction:.1f}% reduction)")
        else:
            print(f"  {model_file}: Quantized version not found")
    
    print("\nONNX Models:")
    for model_file in onnx_models:
        base_name = os.path.splitext(model_file)[0]
        orig_path = os.path.join(models_dir, model_file)
        quant_path = os.path.join(quantized_dir, f"{base_name}_int4.onnx")
        
        if os.path.exists(quant_path):
            orig_size = get_file_size_mb(orig_path)
            quant_size = get_file_size_mb(quant_path)
            reduction = 100 * (1 - quant_size/orig_size)
            print(f"  {model_file}: {orig_size:.2f} MB → {quant_size:.2f} MB ({reduction:.1f}% reduction)")
        else:
            print(f"  {model_file}: Quantized version not found")

def verify_all_models_safe(models_dir="models", quantized_dir="models/quantized"):
    """Safely verify all quantized models"""
    
    # Start with simple file comparison
    simple_file_comparison(models_dir, quantized_dir)
    
    # Ask user if they want detailed analysis
    print("\n" + "="*60)
    response = input("Do you want detailed model analysis? (y/N): ").lower().strip()
    
    if response != 'y':
        print("Skipping detailed analysis to avoid memory issues.")
        return
    
    # Find models
    pytorch_models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    onnx_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
    
    # Verify PyTorch models one by one
    print("\n===== DETAILED PYTORCH MODEL ANALYSIS =====")
    for model_file in pytorch_models:
        base_name = os.path.splitext(model_file)[0]
        orig_path = os.path.join(models_dir, model_file)
        quant_path = os.path.join(quantized_dir, f"{base_name}_int4.pth")
        
        if os.path.exists(quant_path):
            try:
                verify_pytorch_quantization_safe(orig_path, quant_path)
            except Exception as e:
                print(f"Error analyzing {model_file}: {e}")
        
        # Force garbage collection between models
        gc.collect()
    
    # Verify ONNX models one by one
    print("\n===== DETAILED ONNX MODEL ANALYSIS =====")
    for model_file in onnx_models:
        base_name = os.path.splitext(model_file)[0]
        orig_path = os.path.join(models_dir, model_file)
        quant_path = os.path.join(quantized_dir, f"{base_name}_int4.onnx")
        
        if os.path.exists(quant_path):
            try:
                verify_onnx_quantization_safe(orig_path, quant_path)
            except Exception as e:
                print(f"Error analyzing {model_file}: {e}")
        
        # Force garbage collection between models
        gc.collect()

if __name__ == "__main__":
    print("Starting safe model quantization verification...")
    verify_all_models_safe()
    print("\nVerification complete!")