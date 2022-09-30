import pyrootutils
import os
import torchvision.transforms as transforms
import timm 
import torch
import os
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import sys


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

import torchvision.transforms as T
import torch.nn.functional as F


from src import utils

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def predictor(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    
    dictt = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(dictt['state_dict'])
    model.eval()


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

    log.info("Starting testing!")


    img = Image.open(cfg.image)


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Resize(32)])

    img = transform(img)
    img = img[np.newaxis, :]


    model.eval()
    predic = model(img)
    label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    val = F.softmax(predic.detach(), dim=1).numpy()[0]
    sorted_values = np.argsort(val)[::-1]
    for i in sorted_values[:3]:
        print(label[i], round(val[i] * 100, 2))

    print("Predicted class: ", label[np.argmax(val)])

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    predictor(cfg)


if __name__ == "__main__":
    main()



