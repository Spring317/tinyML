from PIL import Image
import numpy as np
import os
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnx
from CustomDataset import CustomDataset
from utilities import manifest_generator_wrapper, get_device

def representative_data_gen(image_size=(160, 160), num_samples=300):

    _, calib, _, _ ,_  = manifest_generator_wrapper(0.3, export=True)
    # print(f"Using calibration data from: {calib}, calib type: {type(calib)}")
    count = 0 
    for fname, _ in calib:
        print(f"Processing file: {fname}")
        img = Image.open(fname).convert("RGB")
        print(f"Image size before resize: {img.size}")
        print(f"image size: {image_size}")
        img = img.resize(image_size, Image.Resampling.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0  # normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, 0)  # add batch dim
        yield {"input": img}
        count +=1 
        if count >= num_samples:
            break
    
class ImageDataReader(CalibrationDataReader):
    def __init__(self, data_gen):
        self.data_iter = iter(data_gen)

    def get_next(self):
        return next(self.data_iter, None)

# Load model
onnx_model_path = "models/mcunet_haute_garonne_other_20.onnx"
quantized_model_path = "models/mcunet_haute_garonne_other_20_species_q8.onnx"

data_reader = ImageDataReader(representative_data_gen(image_size=(160, 160), num_samples=300))
print(f"Quantizing model {onnx_model_path} to {quantized_model_path} using 300 samples for calibration...")
quantize_static(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QDQ,  # or QuantType.QLinearOps for better compatibility
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    per_channel=True,
)
