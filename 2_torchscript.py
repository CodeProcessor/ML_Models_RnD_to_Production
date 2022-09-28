"""
Torch Script
"""

import torch

from params import predict_image

device = "cpu"
model = torch.jit.load('assets/model_scripted.pt', map_location=device)
model.eval()

predict_image(model, "assets/test_image.jpg", o_device=device)
