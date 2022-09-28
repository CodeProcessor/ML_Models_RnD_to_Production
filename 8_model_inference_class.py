import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms

from params import timeit


class MyCustomModel:
    def __init__(self):
        self.device = "cpu"
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.threshold = 0.5
        self.classes = [
            "Black-grass",
            "Charlock"
        ]

    @timeit
    def _load_model(self):
        model = torch.jit.load('assets/model_scripted.pt', map_location=self.device)
        model.eval()
        return model

    def predict(self, image):
        image = self._pre_process(image)
        pred = self._inference(image)
        _class, _confidence = self._post_process(pred)
        return _class, _confidence

    def _pre_process(self, image):
        pil_image = Image.fromarray(image)
        return self.transform(pil_image)

    @timeit
    def _inference(self, image):
        image_device = image.to(self.device)
        image_unz = image_device.unsqueeze(0)
        pred = self.model(image_unz)
        softmax = nn.Softmax(dim=1)
        output = softmax(pred)
        ret = output.detach().cpu().numpy()
        return ret

    def _post_process(self, pred):
        pred = np.squeeze(pred)
        arg_max = np.argmax(pred)
        _confidence = pred[arg_max]
        if _confidence > self.threshold:
            _class = self.classes[arg_max]
        else:
            _class = "Unknown"
            _confidence = 0
        return _class, _confidence


if __name__ == '__main__':
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # model = MyCustomModel()
    # print(model.predict(image))

    # what if this is initialized by two times
    model = MyCustomModel()
    model2 = MyCustomModel()
    print(model.predict(image))
    print(model2.predict(image))