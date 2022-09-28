"""
Save the model after initializing it \
"""

import torch

from model import TheModelClass
from params import device

# Initialize model
model = TheModelClass().to(device)

"""
Train the model
"""

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

PATH = "assets/custom_model_state_dict.pth"
torch.save(model.state_dict(), PATH)

PATH = "assets/custom_model_full.pth"
torch.save(model, PATH)
