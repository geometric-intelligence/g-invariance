import collections

import numpy as np
import torch
from skimage.transform import rotate
import torchvision.transforms.v2 as vt
import math


class Transform:
    def __init__(self):
        self.name = None

    def define_containers(self, tlabels):
        transformed_data, transforms, new_labels = [], [], []
        new_tlabels = collections.OrderedDict({k: [] for k in tlabels.keys()})
        return transformed_data, new_labels, new_tlabels, transforms

    def reformat(self, transformed_data, new_labels, new_tlabels, transforms):
        try:
            transformed_data = torch.stack(transformed_data)
        except:
            transformed_data = torch.tensor(np.array(transformed_data))
        transforms = torch.tensor(transforms)
        new_labels = torch.stack(new_labels)
        for k in new_tlabels.keys():
            new_tlabels[k] = torch.stack(new_tlabels[k])
        return transformed_data, new_labels, new_tlabels, transforms


class AddChannelDim(Transform):
    def __init__(self):
        super().__init__()
        self.name = "add-channel-dim"

    def __call__(self, data, labels, tlabels):
        transformed_data = torch.unsqueeze(data, 1)
        transforms = torch.zeros(len(data))
        return transformed_data, labels, tlabels, transforms


class Transform:
    def __init__(self):
        self.name = None

    def define_containers(self, tlabels):
        transformed_data, transforms, new_labels = [], [], []
        new_tlabels = collections.OrderedDict({k: [] for k in tlabels.keys()})
        return transformed_data, new_labels, new_tlabels, transforms

    def reformat(self, transformed_data, new_labels, new_tlabels, transforms):
        try:
            transformed_data = torch.stack(transformed_data)
        except:
            transformed_data = torch.tensor(np.array(transformed_data))
        transforms = torch.tensor(transforms)
        new_labels = torch.stack(new_labels)
        for k in new_tlabels.keys():
            new_tlabels[k] = torch.stack(new_tlabels[k])
        return transformed_data, new_labels, new_tlabels, transforms


class CircleCrop(Transform):
    def __init__(self):
        super().__init__()
        self.name = "circle-crop"

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        img_size = data.shape[1:]

        v, h = np.mgrid[: img_size[0], : img_size[1]]
        equation = (v - ((img_size[0] - 1) / 2)) ** 2 + (
            h - ((img_size[1] - 1) / 2)
        ) ** 2
        circle = equation < (equation.max() / 2)

        transformed_data = data.clone()
        transformed_data[:, ~circle] = 0.0
        transforms = torch.zeros(len(data))

        return transformed_data, labels, tlabels, transforms


class O2(Transform):
    def __init__(self, n=1, sample_method="linspace"):
        """
        If sample_method == "linspace", rotations will be linspaced. However, flips will be randomized.
        """
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.n = n
        self.sample_method = sample_method
        self.name = "so2"

    def get_transforms(self):
        rot = 360 / self.n
        if self.sample_method == "linspace":
            rot_list = np.array([rot * i for i in range(self.n)])
            rotations = np.hstack([rot_list, rot_list])
            flips = np.hstack([np.zeros(self.n), np.ones(self.n)])
        else:
            rotations = np.random.choice(np.arange(360), size=self.n, replace=False)
            flips = np.random.randint(low=0, high=2, size=(self.n,))
        return rotations, flips

    def get_flips(self):
        if self.sample_method == "linspace":
            n_transforms = int(self.n)
            flips = np.hstack([np.zeros(n_transforms), np.ones(n_transforms)])
        else:
            flips = np.random.randint(low=0, high=2, size=(n_transforms,))
        return flips

    def __call__(self, data, labels, tlabels):
        assert (
            len(data.shape) == 3
        ), "Data must have shape (n_datapoints, img_size[0], img_size[1])"

        transformed_data, new_labels, new_tlabels, transforms = self.define_containers(
            tlabels
        )

        for i, x in enumerate(data):
            rotations, flips = self.get_transforms()
            for j in range(len(rotations)):
                if bool(flips[j]):
                    x_flip = torch.flip(x, dims=(0,))
                else:
                    x_flip = x
                # TODO: use https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.rotate.html?
                xt = rotate(x_flip, rotations[j], interpolation="bilinear")
                transformed_data.append(xt)
                transforms.append((flips[j], rotations[j]))
                new_labels.append(labels[i])
                for k in new_tlabels.keys():
                    new_tlabels[k].append(tlabels[k][i])

        transformed_data, new_labels, new_tlabels, transforms = self.reformat(
            transformed_data, new_labels, new_tlabels, transforms
        )
        return transformed_data, new_labels, new_tlabels, transforms


def resize_image(img, target_size):
    _, size, _ = img.shape

    pad_size = max(target_size - size, 0)

    pad_top = pad_size // 2
    pad_bottom = pad_size - pad_top
    padded_tensor = vt.functional.pad(
        img,
        padding=(pad_top, pad_bottom, pad_bottom, pad_top),
        fill=0,
        padding_mode="constant",
    )
    return padded_tensor


class SO2(torch.nn.Module):
    def __init__(self, n=1, sample_method="linspace"):
        super().__init__()
        assert sample_method in [
            "linspace",
            "random",
        ], "sample_method must be one of ['linspace', 'random']"
        self.n = n
        self.sample_method = sample_method
        self.name = "so2"
        self.sqrt2 = math.sqrt(2)

    def get_samples(self):
        rot = 360 / self.n
        if self.sample_method == "linspace":
            rotations = [rot * i for i in range(self.n)]
        else:
            # TODO: Remove numpy deps
            rotations = np.random.choice(np.arange(360), size=self.n, replace=False)
        return rotations

    def forward(self, x):
        N = x.shape[0]
        target_size = math.ceil(self.sqrt2 * x.shape[-1]) + 1
        processed_images = []
        for i in range(N):
            angles = self.get_samples()
            rotated_images = [
                vt.functional.rotate(x[i : i + 1], angle, expand=True)
                for angle in angles
            ]
            resized_images = [
                resize_image(rotated_images[i], target_size=target_size)
                for i in range(self.n)
            ]
            concatenated_image = torch.stack(resized_images, dim=0)
            processed_images.append(concatenated_image)

        x = torch.cat(processed_images, dim=0)
        return x
