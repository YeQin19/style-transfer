import torch.optim as optim
import torch.nn as nn

def LBFGS(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer