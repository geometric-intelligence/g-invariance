import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError

import numpy as np
import torch
from PIL import Image

from .utils import (
    download_and_extract_archive,
    extract_archive,
    verify_str_arg,
    check_integrity,
)
from .vision import VisionDataset
from torchvision import datasets


class AugmentedMNIST(VisionDataset):
    """Augmented mnist"""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        group: str = "so2",
        n_samples: int = 10, 
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        if file exists:
            self.data = np.load(root + "mnist_augmented.npz")
            self.targets = np.load(root + "mnist_augmented_targets.npz")
        mnist = datasets.MNIST(root, train=train)

        for i in enumerate(mnist):
            rotate...
            append...


        np.save(self.data, root + "mnist_augmented.npz")
        np.save(self.targets, root + "mnist_augmented_targets.npz")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self) -> int:
        return len(self.data)