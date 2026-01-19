import torch
import torch.nn as nn
import torch.nn.functional as F
from training.mask_classification_loss import MaskClassificationLoss

class LogitNormMaskClassificationLoss(MaskClassificationLoss):
    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def loss_labels(self, class_queries_logits, class_labels, indices):
        """
        Compute the class loss using Logit Normalization.
        """
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        
        # Target construction (reusing Mask2Former logic)
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o

        # Logit Normalization
        # pred_logits: (B, N, C)
        # Normalize along class dimension (last dim)
        norms = torch.norm(pred_logits, p=2, dim=-1, keepdim=True) + 1e-7
        logits_norm = torch.div(pred_logits, norms) / self.temperature
        
        # Transpose for CrossEntropy: (B, C, N)
        pred_logits_transposed = logits_norm.transpose(1, 2)
        
        loss_ce = F.cross_entropy(
            pred_logits_transposed, 
            target_classes, 
            weight=self.empty_weight,
            ignore_index=-1 
        )
        
        losses = {"loss_cross_entropy": loss_ce}
        return losses

