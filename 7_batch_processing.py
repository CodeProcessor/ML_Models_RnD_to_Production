import time

import torch

from params import transform
from PIL import Image
device = "cpu"


model = torch.jit.load('assets/model_scripted.pt', map_location=device)
model.eval()

# bath processing
images = [Image.open("assets/test_image.jpg").convert('RGB') for _ in range(20)]
input_batch = torch.stack([transform(image) for image in images])
input_batch = input_batch.to(device)
ts = time.time()
pred = [model(input_batch) for _ in range(5)]
te = time.time()
print(f'predict {round(te - ts, 5)} sec')