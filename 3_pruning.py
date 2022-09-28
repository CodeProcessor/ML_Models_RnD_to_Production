"""
Model pruning - https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
"""
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from model import TheModelClass
from params import device, predict, transform, predict_image

# Initialize model
model = TheModelClass().to(device)
model.eval()

predict_image(model, "assets/test_image.jpg")

# Prune model - Local
module = model.conv1
print(list(module.named_parameters()))
print(list(module.named_buffers()))

print("================")
prune.random_unstructured(module, name="weight", amount=0.3)
print(list(module.named_parameters()))
print(list(module.named_buffers()))

print(module.weight)

prune.remove(module, 'weight')
print(list(module.named_parameters()))

"""
Prune model global
"""
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    )
)
print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    )
)
print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc2.weight == 0))
        / float(model.fc2.weight.nelement())
    )
)
print(
    "Sparsity in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc3.weight == 0))
        / float(model.fc3.weight.nelement())
    )
)
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)



predict_image(model, "assets/test_image.jpg")

PATH = "assets/custom_model_pruned_full.pth"
torch.save(model, PATH)