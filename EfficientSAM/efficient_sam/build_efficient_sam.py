# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .efficient_sam import build_efficient_sam

# Resolve weights path relative to this file's location
_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")

def build_efficient_sam_vitt(img_size=None):
    model = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=os.path.join(_WEIGHTS_DIR, "efficient_sam_vitt.pt"),
    ).eval()
    if img_size is not None and img_size != model.image_encoder.img_size:
        model.image_encoder.img_size = img_size
    return model


def build_efficient_sam_vits(img_size=None):
    model = build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=os.path.join(_WEIGHTS_DIR, "efficient_sam_vits.pt"),
    ).eval()
    if img_size is not None and img_size != model.image_encoder.img_size:
        model.image_encoder.img_size = img_size
    return model
