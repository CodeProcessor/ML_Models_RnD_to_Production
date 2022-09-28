"""
Load the model and run inference
"""

import torch

# Load state dict
from model import TheModelClass
from params import device, predict_image

# Create the model
model = TheModelClass().to(device)
# Load model state dictionary
model.load_state_dict(torch.load("assets/custom_model_state_dict.pth"))

# Load full model
PATH = "assets/custom_model_full.pth"
# Model class must be defined somewhere
model = torch.load(PATH)

# Set model to evaluation mode
model.eval()

predict_image(model, "assets/test_image.jpg")

# Export as torch script
model_scripted = torch.jit.script(model)  # Export to TorchScript
model_scripted.save('assets/model_scripted.pt')  # Save
