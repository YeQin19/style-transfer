from __future__ import print_function

import matplotlib.pyplot as plt
import os

import torch
import torchvision.models as models

from ptnst.loader import Loader
from ptnst.model import get_style_model_and_losses
from ptnst.optimizer import LBFGS
from ptnst.utils import imshow

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def run_style_transfer(content_img, style_img, input_img, device,
                       normalization_mean, normalization_std, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    model, style_losses, content_losses = get_style_model_and_losses(
        style_img, content_img, normalization_mean, normalization_std, vgg)
    model.to(device)
    optimizer = LBFGS(input_img)

    print('Optimizing..')
    epoch = [0]
    while epoch[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print("epoch {}:".format(epoch))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # desired size of the output image
    im_size = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    loader = Loader(im_size)
    style_img = loader.load_img("./img/style_img/picasso.jpg", title='Style Image')
    style_img = style_img.to(torch.device('cuda'))
    content_img = loader.load_img("./img/content_img/dancing.jpg", title='Content Image')
    content_img = content_img.to(torch.device('cuda'))

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    input_img = content_img.clone()

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = run_style_transfer(content_img, style_img, input_img, device,
                                normalization_mean, normalization_std)

    plt.figure()
    imshow(output, title='Output Image')

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()


