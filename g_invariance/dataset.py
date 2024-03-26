import os
import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.transform import rotate
from torchvision import datasets
from IPython import embed


class AugmentedDataset(datasets.VisionDataset):
    """Augmented dataset"""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        sampling_method: str = 'random',
        dataset_name='MNIST',
        group: str = 'so2',
        n_samples: int = 2,
        circle_crop: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.n_samples = n_samples
        if sampling_method not in ['linspace', 'random']:
            raise ValueError("sampling_method must be one of ['linspace', 'random']")
        self.sampling_method = sampling_method
        train_str = 'train' if train else 'val'
        circle_crop_str = '_circle_crop' if circle_crop else ''
        filename_suffix = f'{group}_{circle_crop_str}{n_samples}_{sampling_method}_{train_str}.npy'
        self._data_path = os.path.join(root, f'{dataset_name}_data_{filename_suffix}')
        self._target_path = os.path.join(root, f'{dataset_name}_labels_{filename_suffix}')
        self.group = group
        if os.path.exists(self._data_path) and os.path.exists(self._target_path):
            self.data = np.load(self._data_path)
            self.targets = np.load(self._target_path)
            return

        allowed_datasets = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10']
        if dataset_name not in allowed_datasets:
            raise ValueError(f'dataset_name must be one of {datasets}')
        kwargs = {}
        if dataset_name == 'EMNIST':
            kwargs = {'split': 'letters'}
        ds = getattr(datasets, dataset_name)(root, train=train, download=True, **kwargs)

        data = []
        targets = []
        print('Data augmentation...')
        img_width = np.array(ds[0][0]).shape[0]
        target_size = int(np.ceil(np.sqrt(2) * img_width) + 1)
        # TODO:joblib parallelize this
        i = 0
        for img, label in ds:
            x = np.array(img)
            rotations, flips = self.get_samples()
            rotated_images = [rotate(x, t, resize=not circle_crop) for t in rotations]
            images = rotated_images
            if self.group == 'o2':
                images = [np.flip(x) if f else x for f, x in zip(flips, rotated_images)]
            if not circle_crop:
                final_images = [self.resize_image(img, target_size=target_size) for img in images]
            else:
                final_images = [self.circle_crop(img) for img in images]

            for x in final_images:
                data.append(x)
                targets.append(label)

        self.data = np.stack(data, axis=0)
        self.targets = np.array(targets)
        if dataset_name == 'EMNIST':
            # EMNIST has 26 classes, but the labels are 1-indexed
            self.targets = self.targets - 1
        if not os.path.exists(root):
            print(f'Creating directory {root}...')
            os.makedirs(root)
        print(f'Saving data to {self._data_path} and {self._target_path}...')
        np.save(self._data_path, self.data)
        np.save(self._target_path, self.targets)

    @staticmethod
    def circle_crop(img):
        # Create a circular mask
        h, w = img.shape[:2]
        center = (int(w / 2), int(h / 2))
        radius = np.min([h, w]) // 2  # Use the smallest dimension for the radius

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius

        # Apply mask
        cropped_img = np.zeros_like(img)
        if img.ndim == 3:
            for i in range(3):
                cropped_img[:, :, i] = img[:, :, i] * mask
        else:
            cropped_img[:, :] = img[:, :] * mask
        return cropped_img

    @staticmethod
    def resize_image(img, target_size):
        # Ensure target size is a tuple for consistency
        if isinstance(target_size, int):
            target_height = target_width = target_size
        else:
            target_height, target_width = target_size

        # Get image dimensions
        height, width = img.shape[:2]

        # Calculate padding for height
        pad_height = max(target_height - height, 0)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        # Calculate padding for width
        pad_width = max(target_width - width, 0)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad only spatial dimensions (height and width)
        if img.ndim == 2:  # Grayscale image (2D)
            pad = ((pad_top, pad_bottom), (pad_left, pad_right))
        else:  # Multi-dimensional image
            pad = ((pad_top, pad_bottom), (pad_left, pad_right)) + ((0, 0),) * (img.ndim - 2)
        # Apply padding
        padded_tensor = np.pad(img, pad_width=pad, mode='constant', constant_values=0)

        return padded_tensor

    def get_samples(self):
        if self.sampling_method == 'linspace':
            rot = 360.0 / self.n_samples
            rotations = np.array([rot * i for i in range(self.n_samples)])
            flips = np.hstack([np.zeros(self.n), np.ones(self.n_samples)])
        else:
            rotations = np.random.choice(np.arange(360), size=self.n_samples, replace=False)
            flips = np.random.randint(low=0, high=2, size=(self.n_samples,))
        return rotations, flips

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # XXX: Probably a better way to handle conversion to int8
        img, target = (self.data[index] * 255).astype('int8'), int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if img.ndim == 2:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
