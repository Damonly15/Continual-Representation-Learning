# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]

class MultiCropTransform:
    """Take V random crops of one image"""

    def __init__(self, transform, num_views):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]


class GlobalLocalCropTransform:
    """Produce n_global large-scale crops followed by n_local small-scale crops.

    Returns a list of length (n_global + n_local). The caller is responsible for
    knowing that the first n_global entries are global views and the rest are local.
    """

    def __init__(self, global_transform, local_transform, n_global, n_local):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, x):
        global_crops = [self.global_transform(x) for _ in range(self.n_global)]
        local_crops = [self.local_transform(x) for _ in range(self.n_local)]
        return global_crops + local_crops

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)