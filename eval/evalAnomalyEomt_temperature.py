# Copyright (c) OpenMMLab. All rights reserved.
import os
# Workaround for Windows OpenMP runtime conflict (libomp vs libiomp5md)
# Set before importing torch/numpy/timm/transformers to take effect
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import glob
import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score, roc_curve, auc
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import InterpolationMode

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "eomt"))

from models.vit import ViT
from models.eomt import EoMT
from training.mask_classification_semantic import MaskClassificationSemantic


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def build_model(args, device):
    img_size = (args.img_height, args.img_width)
    encoder = ViT(
        img_size=img_size,
        patch_size=args.patch_size,
        backbone_name=args.backbone_name,
        ckpt_path=args.ckpt_path,
    )
    network = EoMT(
        encoder=encoder,
        num_classes=args.num_classes,
        num_q=args.num_queries,
        num_blocks=args.num_blocks,
        masked_attn_enabled=not args.disable_masked_attn,
    )

    model = MaskClassificationSemantic(
        network=network,
        img_size=img_size,
        num_classes=args.num_classes,
        attn_mask_annealing_enabled=False,
        attn_mask_annealing_start_steps=None,
        attn_mask_annealing_end_steps=None,
        ckpt_path=args.ckpt_path,
        delta_weights=False,
        load_ckpt_class_head=not args.skip_class_head,
    )

    model.eval()
    model.to(device)
    return model

def compute_scores(per_pixel_logits: torch.Tensor):
    temperatures = [0.5, 0.75, 1.0, 1.1]
    msp_scores = {}
    
    for t in temperatures:
        # Scale logits by temperature
        scaled_logits = per_pixel_logits / t
        probs = F.softmax(scaled_logits, dim=0)
        msp = 1.0 - torch.max(probs, dim=0).values
        msp_scores[t] = msp.cpu().numpy()
        
    return msp_scores


def load_mask(path: str):
    mask = Image.open(path)
    #mask = target_transform(mask)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in path:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if "LostAndFound" in path:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in path:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

    return ood_gts

def fpr_at_95_tpr(preds, labels, pos_label=1):
    """Return the FPR when TPR is at minimum 95%.
        
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default=["../Validation_Dataset/RoadAnomaly21/images/*.png"],
        nargs="+",
        help="Glob pattern or list of input images",
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to EoMT checkpoint (.ckpt or .bin)")

    parser.add_argument("--img_height", type=int, default=1024)
    parser.add_argument("--img_width", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--backbone_name", default="vit_base_patch14_reg4_dinov2")
    parser.add_argument("--skip_class_head", action="store_true", help="Skip loading classification head from checkpoint")
    parser.add_argument("--disable_masked_attn", action="store_true", help="Disable masked attention at inference")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--result_file", type=str, default="results_eomt_temp.txt", help="path to result file")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if not os.path.exists(args.result_file):
        open(args.result_file, "w").close()
    file = open(args.result_file, "a")

    model = build_model(args, device)

    # Dictionary to store lists of scores for each temperature
    # Keys will be float temperatures: 0.5, 0.75, 1.0, 1.1
    # Values will be lists of per-image MSP scores (flattened later if needed, but here we append arrays)
    # Actually we append the processed score map usually.
    # In original code lists stored numpy arrays of scores.
    # We will maintain a dict of lists.
    temperatures = [0.5, 0.75, 1.0, 1.1]
    score_lists = {t: [] for t in temperatures}
    ood_gts_list = []

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        # Load full resolution image directly
        pil_img = Image.open(path).convert("RGB")

        # Convert to tensor [C, H, W] in uint8 [0, 255]
        # np.array(pil_img) gives [H, W, C]
        img_uint8 = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(device)
        img_sizes = [img_uint8.shape[-2:]]
        
        imgs = [img_uint8] 
        
        with torch.no_grad():
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)
        
            # We want the last layer result
            mask_logits = mask_logits_per_layer[-1]
            class_logits = class_logits_per_layer[-1]
            mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
            crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            
            # revert_window_logits_semantic returns a list of rebuilt logits [C, H, W]
            logits_list = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            per_pixel_logits = logits_list[0] # Take the first (and only) image
            
        msp_scores_map = compute_scores(per_pixel_logits)

        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        ood_gts = load_mask(pathGT)

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        
        for t in temperatures:
            score_lists[t].append(msp_scores_map[t])

        del mask_logits, class_logits, per_pixel_logits
        torch.cuda.empty_cache()

    file.write("\n")

    if not ood_gts_list:
        print("No OOD pixels found in provided dataset.")
        file.close()
        return

    ood_gts = np.array(ood_gts_list)
    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    file.write(f"{args.ckpt_path}\n")
    file.write(f"{args.input[0]}\n")
    
    for t in temperatures:
        scores = np.array(score_lists[t])
        
        ood_out = scores[ood_mask]
        ind_out = scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        print(f"[MSP T={t}] AUPRC score: {prc_auc * 100.0}")
        print(f"[MSP T={t}] FPR@95TPR: {fpr * 100.0}")

        file.write(
            f"    [MSP T={t}] AUPRC score:{prc_auc * 100.0}   FPR@95TPR:{fpr * 100.0}"
        )

    file.close()


if __name__ == "__main__":
    main()