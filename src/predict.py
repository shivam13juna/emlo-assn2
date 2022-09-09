from typing import Any
from cog import BasePredictor, Input, Path
import pyrootutils
import json
import torch
import timm
import numpy as np
import torchvision.transforms as transforms
import os
from timm.data.transforms_factory import transforms_imagenet_eval

from PIL import Image

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_pylogger(__name__)


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = timm.create_model('efficientnet_b3a', pretrained=True)
        # self.model.eval()
        # self.transform = transforms_imagenet_eval()

        # with open("imagenet_1k.json", "r") as f:
        #     self.labels = list(json.load(f).values())

        self.model = timm.create_model('resnet18', pretrained=True, num_classes = 10)
        dictt = torch.load('epoch_008.ckpt', map_location=torch.device('cpu'))
        state_dict = {k.partition('net.')[2]: v for k,v in dictt['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model"""

        # Preprocess the image
        img = Image.open(image)

        transform = transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor()])

        img = transform(img)
        img = img[np.newaxis, :]


        # print("Predicting and shape of image: ", img.shape)
        # predictions = trainer.predict(model=model, dataloaders=img, ckpt_path=cfg.ckpt_path)
        with torch.no_grad():
            predic = self.model(img)
        label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        val = np.argmax(predic, axis=1).numpy() - 1

        return label[int(val)]




