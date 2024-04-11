import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from experiments.base_exp import Exp
from experiments.vgg_face2_exp import VGGFace2Exp

if __name__ == "__main__":
    conf_cls = VGGFace2Exp()
    trainer = conf_cls.get_trainer()
    trainer.train()








