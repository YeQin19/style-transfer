from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torchvision import transforms


class Loader(data.Dataset):

    def __init__(self,img_size):
        self.img_size = img_size
        self.normMean = [0.485, 0.456, 0.406]
        self.normStd = [0.229, 0.224, 0.225]

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(self.normMean, self.normStd),
            ]
        )

    def load_img(self, img_path, title):
        img = Image.open(img_path).convert('RGB')
        # plt.figure()
        # plt.imshow(img)
        # if title is not None:
        #     plt.title(title)
        # plt.pause(0.001)

        if img.size == self.img_size:
            pass
        else:
            img = img.resize((self.img_size, self.img_size))  # uint8 with RGB mode

        img = self.tf(img)
        img = img.unsqueeze(0)
        img.to(torch.float)
        return img

