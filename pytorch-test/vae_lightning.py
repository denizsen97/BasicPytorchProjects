import pytorch_lightning as pl
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision

class VAE(pl.LightningModule):

    def __init__(self):
        super(VAE, self).__init__()
