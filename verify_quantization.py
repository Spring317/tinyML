import os
import torch
import onnx
import numpy as np
from collections import Counter

def get_file_size_mb(file_path):
    """Get file size in MB"""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def verify_pytorch_quantization(original_path, quantized_path):
    """Verify if PyTorch model is actually quantized"""
    print(f"\n----- PyTorch Model: {os.path.basename(quantized_path)} -----")
    
    # Check file sizes
    orig_size = get_file_size_mb(original_path)
    quant_size = get_file_size_mb(quantized_path)
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Size reduction: {100 * (1 - quant_size/orig_size):.1f}%")
    
    # Load models
    orig_model = torch.load(original_path, weights_only=False)
    quant_model = torch.load(quantized_path, weights_only=False)
    
    # Check parameter types
    orig_dtypes = Counter()
    quant_dtypes = Counter()
    
    for name, param in orig_model.state_dict().items():
        orig_dtypes[str(param.dtype)] += 1
    
    for name, param in quant_model.state_dict().items():
        quant_dtypes[str(param.dtype)] += 1
    
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
        print("The dynamic quantization might only be applied during inference.")

def verify_onnx_quantization(original_path, quantized_path):
    """Verify if ONNX model is actually quantized"""
    print(f"\n----- ONNX Model: {os.path.basename(quantized_path)} -----")
    
    # Check file sizes
    orig_size = get_file_size_mb(original_path)
    quant_size = get_file_size_mb(quantized_path)
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Size reduction: {100 * (1 - quant_size/orig_size):.1f}%")
    
    try:
        # Load models
        orig_model = onnx.load(original_path)
        quant_model = onnx.load(quantized_path)
        
        # Check for quantization nodes
        quant_nodes = ['QuantizeLinear', 'DequantizeLinear']
        
        orig_ops = Counter(node.op_type for node in orig_model.graph.node)
        quant_ops = Counter(node.op_type for node in quant_model.graph.node)
        
        print("\nOriginal model operation types:")
        for op, count in orig_ops.most_common():
            print(f"  {op}: {count}")
        
        print("\nQuantized model operation types:")
        for op, count in quant_ops.most_common():
            print(f"  {op}: {count}")
        
        # Check initializers for INT8 tensors
        orig_init_types = Counter()
        quant_init_types = Counter()
        
        for init in orig_model.graph.initializer:
            orig_init_types[str(init.data_type)] += 1
        
        for init in quant_model.graph.initializer:
            quant_init_types[str(init.data_type)] += 1
        
        print("\nOriginal model tensor types:")
        for dtype, count in orig_init_types.items():
            print(f"  {dtype}: {count}")
        
        print("\nQuantized model tensor types:")
        for dtype, count in quant_init_types.items():
            print(f"  {dtype}: {count}")
        
        # Check if model has quantization nodes or INT8 tensors
        has_quant_ops = any(op_name in quant_ops for op_name in quant_nodes)
        has_int8_tensors = '2' in quant_init_types  # 2 is INT8 in ONNX
        
        if has_quant_ops or has_int8_tensors:
            print("\n✅ VERIFIED: Model contains quantization operations or INT8 tensors")
        else:
            print("\n❌ NOT VERIFIED: Model does not contain quantization operations")
    except Exception as e:
        print(f"Error analyzing ONNX model: {e}")

def verify_all_models(models_dir="models", quantized_dir="models/quantized"):
    """Verify all quantized models"""
    # Find all original models
    pytorch_models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    onnx_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
    
    # Verify PyTorch models
    print("===== VERIFYING PYTORCH MODELS =====")
    for model_file in pytorch_models:
        base_name = os.path.splitext(model_file)[0]
        orig_path = os.path.join(models_dir, model_file)
        quant_path = os.path.join(quantized_dir, f"{base_name}_int4.pth")
        
        if os.path.exists(quant_path):
            verify_pytorch_quantization(orig_path, quant_path)
        else:
            print(f"\n----- PyTorch Model: {model_file} -----")
            print(f"Quantized version not found: {quant_path}")
    
    # Verify ONNX models
    print("\n===== VERIFYING ONNX MODELS =====")
    for model_file in onnx_models:
        base_name = os.path.splitext(model_file)[0]
        orig_path = os.path.join(models_dir, model_file)
        quant_path = os.path.join(quantized_dir, f"{base_name}_int4.onnx")
        
        if os.path.exists(quant_path):
            verify_onnx_quantization(orig_path, quant_path)
        else:
            print(f"\n----- ONNX Model: {model_file} -----")
            print(f"Quantized version not found: {quant_path}")

if __name__ == "__main__":
    print("Starting model quantization verification...")
    verify_all_models()
    print("\nVerification complete!")