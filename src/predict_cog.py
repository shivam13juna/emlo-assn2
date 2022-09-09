from typing import Any
from cog import BasePredictor, Input, Path
import pyrootutils
import json
import torch
import timm
import numpy as np
import torchvision.transforms as transforms

from timm.data.transforms_factory import transforms_imagenet_eval

from PIL import Image

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
from src.predict import predictor_cog

log = utils.get_pylogger(__name__)


root = pyrootutils.setup_root(
    search_from=os.getcwd(),
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = timm.create_model('efficientnet_b3a', pretrained=True)
        self.model.eval()
        self.transform = transforms_imagenet_eval()

        with open("imagenet_1k.json", "r") as f:
            self.labels = list(json.load(f).values())

    # Define the arguments and types the model takes as input
    @utils.task_wrapper
    def predict(self, cfg, image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model"""
        assert cfg.ckpt_path

        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("Instantiating loggers...")
        logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters(object_dict)


        # Preprocess the image
        img = Image.open(cfg.image)
        print("Got response")

        print("Creating Transform")


        transform = transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor()])

        img = transform(img)

        # img = img[-1]
        # img = img[np.newaxis, :]
        img = img[np.newaxis, :]


        print("Predicting and shape of image: ", img.shape)
        # predictions = trainer.predict(model=model, dataloaders=img, ckpt_path=cfg.ckpt_path)
        model.eval()
        with torch.no_grad():
            predic = model(img)
        label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        val = np.argmax(predic, axis=1).numpy() - 1
        print("val: ", val)
        print("This is the prediction? : ", predic.shape, np.round(predic, 2), label[int(val)])

        return label[int(val)]



@hydra.main(version_base="1.2", config_path=root / "configs", config_name="predict.yaml")
def main(cfg: DictConfig, image) -> None:
    predictor_cog(cfg, image)


