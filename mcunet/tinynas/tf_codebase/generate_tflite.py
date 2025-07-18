import tensorflow as tf
import torch
import functools
import numpy as np


def generate_tflite_with_weight(pt_model, resolution, tflite_fname, calib_loader,
                                n_calibrate_sample=500):
    # 1. convert the state_dict to tensorflow format
    pt_sd = pt_model.state_dict()

    tf_sd = {}
    for key, v in pt_sd.items():
        if key.endswith('depth_conv.conv.weight'):
            v = v.permute(2, 3, 0, 1)
        elif key.endswith('conv.weight'):
            v = v.permute(2, 3, 1, 0)
        elif key == 'classifier.linear.weight':
            v = v.permute(1, 0)
        tf_sd[key.replace('.', '/')] = v.numpy()

    # 2. build the tf network using the same config
    weight_decay = 0.

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            def network_map(images):
                net_config = pt_model.config
                from .tf_modules import ProxylessNASNets
                net_tf = ProxylessNASNets(net_config=net_config, net_weights=tf_sd,
                                          n_classes=pt_model.classifier.linear.out_features,
                                          graph=graph, sess=sess, is_training=False,
                                          images=images, img_size=resolution)
                logits = net_tf.logits
                return logits, {}

            def arg_scopes_map(weight_decay=0.):
                arg_scope = tf.contrib.framework.arg_scope
                with arg_scope([]) as sc:
                    return sc

            slim = tf.contrib.slim

            @functools.wraps(network_map)
            def network_fn(images):
                arg_scope = arg_scopes_map(weight_decay=weight_decay)
                with slim.arg_scope(arg_scope):
                    return network_map(images)

            input_shape = [1, resolution, resolution, 3]
            placeholder = tf.placeholder(name='input', dtype=tf.float32, shape=input_shape)

            out, _ = network_fn(placeholder)

            # 3. convert to tflite (with int8 quantization)
            converter = tf.lite.TFLiteConverter.from_session(sess, [placeholder], [out])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.inference_output_type = tf.int8
            converter.inference_input_type = tf.int8

            def representative_dataset_gen():
                for i_b, (data, _) in enumerate(calib_loader):
                    if i_b == n_calibrate_sample:
                        break
                    data = data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
                    yield [data]

            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            tflite_buffer = converter.convert()
            tf.gfile.GFile(tflite_fname, "wb").write(tflite_buffer)


if __name__ == '__main__':
    import sys
    import json
    import os
    
    # Updated argument parsing to work with train.py models
    if len(sys.argv) != 3:
        print("Usage: python generate_tflite.py <model_path.pth> <output.tflite>")
        print("Example: python generate_tflite.py models/mcunet-in2_haute_garonne_10_species.pth models/output.tflite")
        sys.exit(1)
    
    model_path = sys.argv[1]  # Path to .pth file from train.py
    tflite_path = sys.argv[2]  # Output .tflite path
    
    # Load the model saved by train.py
    print(f"Loading model from: {model_path}")
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    print(f"Model loaded successfully: {model_path}")
    # Extract model configuration and resolution
    # For MCUNet models, we need to infer the resolution from the model name or config
    model_name = os.path.basename(model_path).split('_')[0]  # Extract mcunet-in2 from filename
    print(f"Model name extracted: {model_name}")
    # Set resolution based on model type (adjust as needed)
    resolution_map = {
        # 'mcunet-in1': 160,
        'mcunet-in2': 160, 
        # 'mcunet-in4': 160,
        # 'mcunet-in5': 160,
        # 'mcunet-in6': 160
    }
    
    resolution = resolution_map.get(model_name, 160)
    print(f"Using resolution: {resolution}x{resolution}")
    
    # Load validation dataset for calibration
    # Try to load the validation dataset saved by train.py
    val_dataset_path = "val_dataset.pt"
    if os.path.exists(val_dataset_path):
        print(f"Loading validation dataset from: {val_dataset_path}")
        val_dataset = torch.load(val_dataset_path, map_location=torch.device('cpu'))
        calib_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=1,
            shuffle=True, 
            num_workers=0  # Set to 0 for compatibility
        )
    else:
        # Fallback: create a simple calibration dataset
        print("No validation dataset found, creating dummy calibration data")
        import torchvision.transforms as transforms
        
        # Create dummy data for calibration
        dummy_data = []
        for i in range(500):
            # Create random tensor with correct shape
            dummy_tensor = torch.randn(3, resolution, resolution)
            dummy_data.append((dummy_tensor, 0))  # (image, dummy_label)
        
        calib_loader = torch.utils.data.DataLoader(
            dummy_data,
            batch_size=1,
            shuffle=False
        )
    
    print(f"Generating TensorFlow Lite model: {tflite_path}")
    
    # Generate the TensorFlow Lite model
    try:
        generate_tflite_with_weight(
            model, 
            resolution, 
            tflite_path, 
            calib_loader,
            n_calibrate_sample=min(500, len(calib_loader))
        )
        print(f"Successfully generated TensorFlow Lite model: {tflite_path}")
        
        # Print file size
        if os.path.exists(tflite_path):
            size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB")
            
    except Exception as e:
        print(f"Error generating TensorFlow Lite model: {e}")
        import traceback
        traceback.print_exc()

