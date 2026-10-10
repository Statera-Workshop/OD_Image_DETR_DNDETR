"""
DN-DETR builder (DN-DAB-DETR, ResNet-50, 300 queries + denoising groups), Image task.

Paper: Li et al., "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising" (CVPR 2022, arXiv 2203.01305).

Code : official IDEA-Research/DN-DETR (models/DN_DAB_DETR) - model files copied unchanged into detectors/dn_detr/official/ except the imports
       (util.* -> detectors/detr/official misc_min + the repository's own box_ops; backbone -> detectors/detr/adapter.py)
       and the [Framework] fixes marked in the files. Built by the repository's own build function with its own default
       arguments (official/args_official.py) through detectors/detr/family.py.
       use_dn True (the DN-DETR method; the repository default without --use_dn trains plain DAB-DETR).

[Unified] no ImageNet weights, trainable BN + PreciseBN, common init (nn.Conv2d / BN only), no mean / std, fixed 608
input without padding masks, Adam 1e-4 for all parameters (official AdamW, backbone 1e-5) with the official gradient
clipping 0.1, 30 epochs, common augmentation and post-processing (NMS on top of the NMS-free set prediction).
"""

from detectors.detr.family import OfficialDETRFamily
from detectors.dn_detr.official.DABDETR import build_DABDETR
from detectors.dn_detr.official.args_official import ARGS


def build_model(task, num_classes):
    return OfficialDETRFamily(num_classes, build_DABDETR, ARGS, call="dn", overrides={'use_dn': True})
