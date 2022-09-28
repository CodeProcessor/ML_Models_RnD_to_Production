# Some standard imports
import time

import torch.onnx

from model import TheModelClass
from params import get_input

"""
Convert model to ONNX format
"""

onnx_model_path = "assets/custom_model.onnx"
batch_size = 1  # just a random number

# Initialize model
device = torch.device("cpu")
torch_model = TheModelClass().to(device)
torch_model.eval()

# Input to the model
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  onnx_model_path,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

"""
ONNX Model inference
"""

import onnxruntime

ort_session = onnxruntime.InferenceSession(onnx_model_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

x = get_input("assets/test_image.jpg", device)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

# Prediction with time
ts = time.time()
ort_outs = [ort_session.run(None, ort_inputs) for _ in range(101)]
te = time.time()
print(f'predict {round(te - ts, 5)} sec')

img_out_y = ort_outs[0]
print(img_out_y)
