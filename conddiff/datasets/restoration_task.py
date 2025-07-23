from itertools import combinations
from typing import Optional

import numpy as np

from task_utils.base_task import BaseTask
from task_utils.deformation_task import BendSinkDeformationTask, BendSourceDeformationTask
from task_utils.intensity_tasks import SmoothNoiseAdditionTask
from task_utils.patch_blending_task import PoissonImageEditingMixedGradBlender, \
    PoissonImageEditingSourceGradBlender
from task_utils.labelling import RestorationLabeller
NUM_TASKS = 5
SAME_DSET_BLEND = 0
OTHER_DSET_BLEND = 1
SINK_DEFORM = 2
SOURCE_DEFORM = 3
INTENSITY_CHANGE = 4



class RestorationTask:

    def __init__(self, has_bg: bool, center: bool, p_no_aug: float, task_kwargs: dict):
        self.has_bg = has_bg
        self.center = center
        self.p_no_aug = p_no_aug
        self.task_kwargs = task_kwargs
        self.task_ids = list(range(NUM_TASKS))  # or a custom list
        self.rng = np.random.default_rng()
        self.res_labeller = RestorationLabeller()

        self.sink_task = BendSinkDeformationTask(
            self.res_labeller, **self.task_kwargs)
        self.source_task = BendSourceDeformationTask(
            self.res_labeller, **self.task_kwargs)
        self.intensity_task = SmoothNoiseAdditionTask(
            self.res_labeller, **self.task_kwargs)

        self.task_map = {
            SAME_DSET_BLEND: self._apply_blending_task,
            OTHER_DSET_BLEND:
                lambda img, pathol_img: self._apply_blending_task(img, pathol_img),
            SINK_DEFORM: lambda img, _: self._apply_task(img, self.sink_task),
            SOURCE_DEFORM: lambda img, _: self._apply_task(img, self.source_task),
            INTENSITY_CHANGE: lambda img, _: self._apply_task(img, self.intensity_task)
        }
  
    
    def apply(self, img: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        if self.rng.random() < self.p_no_aug:
            orig_img = img
            if self.center:
                orig_img = (orig_img - 0.5) * 2
            return img, orig_img

        corrupted_img, original_img = self.task_map[self.rng.choice(self.task_ids)](img, path)

        # The corrupted image will be centered by the dataset, as it's the model input.
        # We need to center the original image as the pipeline doesn't know it's an image and not a label
        if self.center:
            original_img = (original_img - 0.5) * 2

        return corrupted_img, original_img

    def __call__(self, img: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.apply(img, path)

    def _maybe_mask_sample(self, img: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        return img, None

    def _apply_task(self, img: np.ndarray, task: BaseTask) -> tuple[np.ndarray[float], np.ndarray[float]]:

        return task(*self._maybe_mask_sample(img), None)

    def _apply_blending_task(self, img: np.ndarray, path: np.ndarray) -> tuple[np.ndarray[float], np.ndarray[float]]:

        # Default to only mixed gradients, as previous work showed mixed to be better than source
        mixed_prop = self.task_kwargs['mixed_prop'] if 'mixed_prop' in self.task_kwargs else 1.

        def get_sample_fn():
            return self._maybe_mask_sample(path)

        if self.rng.random() < mixed_prop:
            blender = PoissonImageEditingMixedGradBlender(
                self.res_labeller, get_sample_fn, **self.task_kwargs)
        else:
            blender = PoissonImageEditingSourceGradBlender(
                self.res_labeller, get_sample_fn, **self.task_kwargs)

        return self._apply_task(img, blender)
