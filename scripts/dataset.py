import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple
from skimage.transform import rotate, resize

import numpy as np

from PIL import Image
from torchvision import datasets


class AugmentedMNIST(datasets.VisionDataset):
    """Augmented mnist"""

    MNIST_SIZE = 28

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        sampling_method: str = "random",
        group: str = "so2",
        n_samples: int = 2,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.n_samples = n_samples
        if sampling_method not in ["linspace", "random"]:
            raise ValueError("sampling_method must be one of ['linspace', 'random']")
        self.sampling_method = sampling_method
        train_str = "train" if train else "t10k"
        filename_suffix = f"{group}_{n_samples}_{sampling_method}_{train_str}.npy"
        self._data_path = os.path.join(root, f"mnist_data_{filename_suffix}")
        self._target_path = os.path.join(root, f"mnist_labels_{filename_suffix}")

        if os.path.exists(self._data_path) and os.path.exists(self._target_path):
            self.data = np.load(self._data_path)
            self.targets = np.load(self._target_path)
            return

        mnist = datasets.MNIST(root, train=train, download=True)

        data = []
        targets = []
        print("generating data augmentation...")

        target_size = int(np.ceil(np.sqrt(2) * self.MNIST_SIZE) + 1)
        # TODO: joblib parallelize this
        for img, label in mnist:
            x = np.array(img)
            rotations = self.get_samples()
            rotated_images = [rotate(x, t, resize=True) for t in rotations]
            resized_images = [
                self.resize_image(img, target_size=target_size)
                for img in rotated_images
            ]
            for x in resized_images:
                data.append(x)
                targets.append(label)
        self.data = np.stack(data, axis=0)
        self.targets = np.array(targets)
        if not os.path.exists(root):
            print(f"Creating directory {root}...")
            os.makedirs(root)
        print(f"Saving data to {self._data_path} and {self._target_path}...")
        np.save(self._data_path, self.data)
        np.save(self._target_path, self.targets)

    @staticmethod
    def resize_image(img, target_size):
        size, _ = img.shape
        pad_size = max(target_size - size, 0)
        pad_top = pad_size // 2
        pad_bottom = pad_size - pad_top
        pad = ((pad_top, pad_bottom), (pad_bottom, pad_top))
        padded_tensor = np.pad(
            img,
            pad_width=pad,
            mode="constant",
            constant_values=((0, 0), (0, 0)),
        )
        return padded_tensor

    def get_samples(self):
        if self.sampling_method == "linspace":
            rot = 360.0 / self.n_samples
            rotations = np.array([rot * i for i in range(self.n_samples)])
        else:
            rotations = np.random.choice(
                np.arange(360), size=self.n_samples, replace=False
            )
        return rotations

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # XXX: Probably a better way to handle conversion to int8
        img, target = (self.data[index] * 255).astype("int8"), int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
