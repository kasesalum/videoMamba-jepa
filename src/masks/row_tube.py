# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from multiprocessing import Value

import torch


class MaskCollator(object):
    """Mask full spatial rows across the temporal tube.

    This is a VideoMamba-style diagnostic mask. For each sample, complete
    patch rows are selected as prediction targets for every time step; the
    remaining rows form the context mask.
    """

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            self.mask_generators.append(
                _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=num_frames,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    ratio=m.get('ratio', m.get('row_ratio', 0.9)),
                    shared_rows_across_batch=m.get(
                        'shared_rows_across_batch', False)))

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):
        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for mask_generator in self.mask_generators:
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        ratio=0.9,
        shared_rows_across_batch=False,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2

        self.height = crop_size[0] // spatial_patch_size[0]
        self.width = crop_size[1] // spatial_patch_size[1]
        self.duration = num_frames // temporal_patch_size
        self.ratio = ratio
        self.shared_rows_across_batch = shared_rows_across_batch
        self.num_mask_rows = max(1, min(
            self.height - 1,
            int(round(self.height * self.ratio))))
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_rows(self, generator=None):
        perm = torch.randperm(self.height, generator=generator)
        return perm[:self.num_mask_rows]

    def _rows_to_masks(self, rows):
        mask = torch.ones(
            (self.duration, self.height, self.width),
            dtype=torch.int32)
        mask[:, rows, :] = 0
        mask = mask.flatten()
        mask_p = torch.argwhere(mask == 0).squeeze()
        mask_e = torch.nonzero(mask).squeeze()
        return mask_e, mask_p

    def __call__(self, batch_size):
        seed = self.step()
        generator = torch.Generator()
        generator.manual_seed(seed)

        shared_rows = None
        if self.shared_rows_across_batch:
            shared_rows = self._sample_rows(generator=generator)

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(batch_size):
            rows = shared_rows
            if rows is None:
                rows = self._sample_rows()
            mask_e, mask_p = self._rows_to_masks(rows)
            collated_masks_enc.append(mask_e)
            collated_masks_pred.append(mask_p)

        return (
            torch.utils.data.default_collate(collated_masks_enc),
            torch.utils.data.default_collate(collated_masks_pred))
