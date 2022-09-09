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
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    print("Getting response")
    # response = requests.get(cfg.image)
    # img = Image.open(BytesIO(response.content))
    img = Image.open(cfg.image)
    print("Got response")

    print("Creating Transform")
    # normalize = transforms.Normalize(mean=list(model.default_cfg['mean']),
    #                                 std=list(model.default_cfg['std']))



    # transform = transforms.Compose([
    #                     transforms.Resize(256),
    #                     transforms.CenterCrop(224),
    #                     transforms.ToTensor(),
    #                     normalize,
    #                     ])


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

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    main()



