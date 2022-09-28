"""
Model Quantization - https://pytorch.org/docs/stable/quantization.html
"""

import torch

from model import TheModelClass
from params import device, predict_image

# Initialize model
device = torch.device("cpu")
model_fp32 = TheModelClass().to(device)

model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

# run the model
# input_fp32 = torch.randn(4, 4, 4, 4)
# res = model_int8(input_fp32)

image_path = "assets/test_image.jpg"
predict_image(model_fp32, image_path, "cpu")
predict_image(model_int8, image_path, "cpu")

# Save the model
PATH = "assets/custom_model_int8.pth"
torch.save(model_int8, PATH)
