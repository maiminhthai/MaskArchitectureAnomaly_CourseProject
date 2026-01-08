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
import math
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


def build_transforms(img_height: int, img_width: int):
    input_t = Compose(
        [
            Resize((img_height, img_width), InterpolationMode.BILINEAR),
            ToTensor(),
        ]
    )
    target_t = Compose([Resize((img_height, img_width), InterpolationMode.NEAREST)])
    return input_t, target_t


def interpolate_pos_embed(state_dict, key, new_size, patch_size):
    """
    Interpolate positional embeddings to match the new image size.
    """
    if key not in state_dict:
        return state_dict

    pos_embed_checkpoint = state_dict[key]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = pos_embed_checkpoint.shape[1]
    
    # Check if we have a class token
    # Assuming square existing grid
    # If num_patches is a perfect square, likely no class token or class token is separate
    # If existing is 4096 -> 64x64
    
    # However, standard ViT might have 1 class token. sqrt(4096) = 64.
    # If it was 4097, it would be 1 + 64x64.
    
    # Based on the error: "torch.Size([1, 4096, 768])", it seems there is NO class token 
    # included in this specific tensor, OR it's a specific architecture without it (like ViTDet or MAE sometimes).
    # We will assume it is purely spatial for now based on the numbers.
    
    n = int(math.sqrt(num_patches))
    if n * n != num_patches:
        # Maybe there is a class token?
        # Try n = sqrt(num_patches - 1)
        n_cls = int(math.sqrt(num_patches - 1))
        if n_cls * n_cls == num_patches - 1:
            has_cls_token = True
            n = n_cls
        else:
            print(f"Warning: Could not determine grid size for {key} with shape {pos_embed_checkpoint.shape}. Skipping interpolation.")
            return state_dict
    else:
        has_cls_token = False

    # Target grid size
    new_h, new_w = new_size
    new_grid_h = new_h // patch_size
    new_grid_w = new_w // patch_size
    
    # If shapes match, return
    if has_cls_token:
        target_num_patches = new_grid_h * new_grid_w + 1
    else:
        target_num_patches = new_grid_h * new_grid_w
        
    if num_patches == target_num_patches:
        return state_dict

    print(f"Interpolating {key} from {num_patches} patches to {target_num_patches} patches...")

    if has_cls_token:
        cls_token = pos_embed_checkpoint[:, 0:1, :]
        pos_embed_grid = pos_embed_checkpoint[:, 1:, :]
    else:
        pos_embed_grid = pos_embed_checkpoint
        
    # Reshape to (B, H, W, C) -> (B, C, H, W) for interpolate
    # We assume square input grid
    pos_embed_grid = pos_embed_grid.reshape(1, n, n, embedding_size).permute(0, 3, 1, 2)
    
    new_pos_embed_grid = F.interpolate(
        pos_embed_grid,
        size=(new_grid_h, new_grid_w),
        mode='bicubic',
        align_corners=False
    )
    
    # Reshape back to (B, N, C)
    new_pos_embed_grid = new_pos_embed_grid.permute(0, 2, 3, 1).reshape(1, -1, embedding_size)
    
    if has_cls_token:
        new_pos_embed = torch.cat((cls_token, new_pos_embed_grid), dim=1)
    else:
        new_pos_embed = new_pos_embed_grid
        
    state_dict[key] = new_pos_embed
    return state_dict


def build_model(args, device):
    img_size = (args.img_height, args.img_width)
    encoder = ViT(
        img_size=img_size,
        patch_size=args.patch_size,
        backbone_name=args.backbone_name,
        ckpt_path=args.encoder_ckpt,
    )
    network = EoMT(
        encoder=encoder,
        num_classes=args.num_classes,
        num_q=args.num_queries,
        num_blocks=args.num_blocks,
        masked_attn_enabled=not args.disable_masked_attn,
    )

    # 1. Initialize model with ckpt_path=None to assume random init first
    model = MaskClassificationSemantic(
        network=network,
        img_size=img_size,
        num_classes=args.num_classes,
        attn_mask_annealing_enabled=False,
        attn_mask_annealing_start_steps=None,
        attn_mask_annealing_end_steps=None,
        ckpt_path=None,  # We load manually
        delta_weights=False,
        load_ckpt_class_head=not args.skip_class_head,
    )

    # 2. Manual loading
    print(f"Loading checkpoint from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 3. Interpolate pos_embed if needed
    # The key in the error was "network.encoder.backbone.pos_embed"
    # But sometimes it might be "model.network..." depending on saving.
    # We look for "pos_embed" keys.
    keys_to_interpolate = [k for k in state_dict.keys() if "pos_embed" in k]
    for key in keys_to_interpolate:
        state_dict = interpolate_pos_embed(
            state_dict, 
            key, 
            new_size=img_size, 
            patch_size=args.patch_size
        )

    # 4. Load state dict
    # We use strict=False to be robust against other minor mismatches, 
    # effectively matching the original script's behavior but with our patched dict.
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model state dict loaded. info: {msg}")

    model.eval()
    model.to(device)
    return model


def to_per_pixel_logits(mask_logits: torch.Tensor, class_logits: torch.Tensor):
    class_probs = class_logits.softmax(dim=-1)[..., :-1]
    return torch.einsum("bqhw,bqc->bchw", mask_logits.sigmoid(), class_probs)


def compute_scores(per_pixel_logits: torch.Tensor):
    probs = F.softmax(per_pixel_logits, dim=0)
    msp = 1.0 - torch.max(probs, dim=0).values
    max_logit = 1.0 - torch.max(per_pixel_logits, dim=0).values
    log_probs = torch.log_softmax(per_pixel_logits, dim=0)
    entropy = -(probs * log_probs).sum(dim=0)
    return msp.cpu().numpy(), max_logit.cpu().numpy(), entropy.cpu().numpy()


def load_mask(path: str, target_transform):
    mask = Image.open(path)
    mask = target_transform(mask)
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
    parser.add_argument(
        "--encoder_ckpt",
        default=None,
        help="Optional encoder checkpoint; leave None to use pretrained backbone weights",
    )
    parser.add_argument("--img_height", type=int, default=640)
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--backbone_name", default="vit_base_patch14_reg4_dinov2")
    parser.add_argument("--skip_class_head", action="store_true", help="Skip loading classification head from checkpoint")
    parser.add_argument("--disable_masked_attn", action="store_true", help="Disable masked attention at inference")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--result_file", default="results_eomt.txt", help="File to append results to")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    input_transform, target_transform = build_transforms(args.img_height, args.img_width)

    if not os.path.exists(args.result_file):
        open(args.result_file, "w").close()
    file = open(args.result_file, "a")

    model = build_model(args, device)

    msp_score_list, max_logit_score_list, max_entropy_score_list, ood_gts_list = [], [], [], []

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        img = input_transform(Image.open(path).convert("RGB"))
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            mask_logits_list, class_logits_list = model.network(img)

        mask_logits = mask_logits_list[-1]
        class_logits = class_logits_list[-1]

        mask_logits = F.interpolate(
            mask_logits,
            size=(args.img_height, args.img_width),
            mode="bilinear",
            align_corners=False,
        )

        per_pixel_logits = to_per_pixel_logits(mask_logits, class_logits).squeeze(0)
        msp_score, max_logit_score, max_entropy_score = compute_scores(per_pixel_logits)

        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        ood_gts = load_mask(pathGT, target_transform)

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        msp_score_list.append(msp_score)
        max_logit_score_list.append(max_logit_score)
        max_entropy_score_list.append(max_entropy_score)

        del mask_logits, class_logits, per_pixel_logits
        torch.cuda.empty_cache()

    file.write("\n")

    if not ood_gts_list:
        print("No OOD pixels found in provided dataset.")
        file.close()
        return

    ood_gts = np.array(ood_gts_list)

    methods = {
        "MSP": np.array(msp_score_list),
        "Max_Logit": np.array(max_logit_score_list),
        "Max_Entropy": np.array(max_entropy_score_list),
    }

    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    file.write(f"{args.ckpt_path}\n")
    file.write(f"{args.input[0]}\n")
    for name, scores in methods.items():
        ood_out = scores[ood_mask]
        ind_out = scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        print(f"[{name}] AUPRC score: {prc_auc * 100.0}")
        print(f"[{name}] FPR@95TPR: {fpr * 100.0}")

        file.write(
            f"    [{name}] AUPRC score:{prc_auc * 100.0}   FPR@95TPR:{fpr * 100.0}"
        )

    file.close()


if __name__ == "__main__":
    main()
