from .shapenet import ShapeNetDataset
from .customized import CustomizedDataset
from .av import AVDataset
from .combined import CombinedDataset
from .points2surf import Points2SurfDataset
from .matterport import MatterportDataset


def build_dataset(name: str, spec, hparams, kwargs: dict):
    return eval(name)(**kwargs, spec=spec, hparams=hparams)
