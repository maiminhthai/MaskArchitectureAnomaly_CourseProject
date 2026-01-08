
import sys
import os
import torch
import unittest

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())
# Also add eomt directory to path if needed for imports within the module to work
sys.path.append(os.path.join(os.getcwd(), 'eomt'))

from eomt.training.logit_norm_loss import LogitNormMaskClassificationLoss
from eomt.training.mask_classification_loss import MaskClassificationLoss

class TestLogitNormMaskClassificationLoss(unittest.TestCase):
    def setUp(self):
        self.num_classes = 19
        self.num_points = 12544
        self.common_args = {
            "num_points": self.num_points,
            "oversample_ratio": 3.0,
            "importance_sample_ratio": 0.75,
            "mask_coefficient": 5.0,
            "dice_coefficient": 5.0,
            "class_coefficient": 2.0,
            "num_labels": self.num_classes,
            "no_object_coefficient": 0.1
        }
        self.loss = LogitNormMaskClassificationLoss(
            temperature=0.01,
            **self.common_args
        )
        
    def test_forward_pass_shapes_and_keys(self):
        batch_size = 2
        num_queries = 10
        h, w = 32, 32
        
        # Mock predicted logits
        # Mask logits: (B, Q, H, W)
        masks_queries_logits = torch.randn(batch_size, num_queries, h, w)
        # Class logits: (B, Q, C + 1) -> +1 for no object
        class_queries_logits = torch.randn(batch_size, num_queries, self.num_classes + 1)
        
        # Mock targets
        # List of dicts with 'masks' and 'labels'
        targets = []
        for _ in range(batch_size):
            num_targets = 3
            # Masks: (N, H, W) binary masks
            masks = torch.randint(0, 2, (num_targets, h, w)).float()
            # Labels: (N,) class indices
            labels = torch.randint(0, self.num_classes, (num_targets,))
            targets.append({
                "masks": masks,
                "labels": labels
            })
            
        # Run forward pass
        loss_dict = self.loss(masks_queries_logits, targets, class_queries_logits)
        
        # Check keys
        print("Loss dict keys:", loss_dict.keys())
        self.assertIn("loss_cross_entropy", loss_dict)
        self.assertIn("loss_mask", loss_dict)
        self.assertIn("loss_dice", loss_dict)
        
        # Check values are scalar tensors
        for k, v in loss_dict.items():
            self.assertTrue(torch.is_tensor(v))
            self.assertEqual(v.numel(), 1)
            self.assertFalse(torch.isnan(v).any(), f"Loss {k} is NaN")
            
    def test_logit_norm_logic(self):
        # Specific check for logit normalization math
        # We invoke loss_labels directly
        
        B, Q = 1, 5
        C = self.num_classes
        pred_logits = torch.randn(B, Q, C + 1)
        # Create dummy indices as returned by matcher: list of (src_idx, tgt_idx) tuples for each batch
        # [(src_indices, tgt_indices)]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))] 
        class_labels = [torch.tensor([0, 1])]
        
        loss_dict = self.loss.loss_labels(pred_logits, class_labels, indices)
        self.assertIn("loss_cross_entropy", loss_dict)
        val = loss_dict["loss_cross_entropy"]
        print(f"Computed LogitNorm CE Loss: {val.item()}")
        
        self.assertTrue(val > 0)

    def test_mask_dice_consistency(self):
        # Verify that loss_mask and loss_dice are identical to parent class
        parent_loss = MaskClassificationLoss(**self.common_args)
        
        batch_size = 2
        num_queries = 5
        h, w = 32, 32
        
        # Same inputs
        masks_queries_logits = torch.randn(batch_size, num_queries, h, w)
        class_queries_logits = torch.randn(batch_size, num_queries, self.num_classes + 1)
        
        targets = []
        for _ in range(batch_size):
            masks = torch.randint(0, 2, (3, h, w)).float()
            labels = torch.randint(0, self.num_classes, (3,))
            targets.append({"masks": masks, "labels": labels})
            
        # Forward passes
        # Matcher uses Hungarian algorithm which is deterministic for same inputs
        # BUT point sampling in loss_masks is random, so we must reset seed.
        
        torch.manual_seed(42)
        res_child = self.loss(masks_queries_logits, targets, class_queries_logits)
        
        torch.manual_seed(42)
        res_parent = parent_loss(masks_queries_logits, targets, class_queries_logits)
        
        self.assertEqual(res_child['loss_mask'].item(), res_parent['loss_mask'].item())
        self.assertEqual(res_child['loss_dice'].item(), res_parent['loss_dice'].item())
        print(f"Consistency Check - Child Mask Loss: {res_child['loss_mask'].item()}, Parent Mask Loss: {res_parent['loss_mask'].item()}")
        print(f"Consistency Check - Child Dice Loss: {res_child['loss_dice'].item()}, Parent Dice Loss: {res_parent['loss_dice'].item()}")

if __name__ == '__main__':
    unittest.main()
