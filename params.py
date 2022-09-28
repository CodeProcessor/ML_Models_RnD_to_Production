import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# Calculate time decorator
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__} {round(te - ts, 5)} sec')
        return result

    return timed


# Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
device = "cuda:0" if torch.cuda.is_available() else "cpu"


@timeit
def predict(model, input):
    [model(input) for _ in range(100)]
    return model(input)


def get_input(image_path, o_device=None):
    pil_image = Image.open(image_path).convert('RGB')
    img = transform(pil_image)
    _device = o_device or device
    inputs = img.to(_device)
    inputs_unz = inputs.unsqueeze(0)
    return inputs_unz

def predict_image(model, image_path, o_device=None):
    inputs_unz = get_input(image_path, o_device)
    pred = predict(model, inputs_unz)
    softmax = nn.Softmax(dim=1)
    output = softmax(pred)
    print(output)
