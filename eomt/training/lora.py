import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

def inject_lora(model, r, lora_alpha, lora_dropout, target_modules):
    """
    Injects LoRA adapters into the model using PEFT.
    
    Args:
        model: The PyTorch model to adapt.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to target. 
    """
    
    existing_modules = []
    for name, module in model.named_modules():
        # Check if any target string is part of the module name
        for target in target_modules:
            if target in name.split('.')[-1]:
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    # Iterate over children to find supported types
                    for child_name, child in module.named_children():
                        if isinstance(child, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                            # Construct the full name relative to the matched module's name provided in LoraConfig
                            existing_modules.append(f"{target}.{child_name}")
                else:
                    existing_modules.append(target)
    
    unique_targets = list(set(existing_modules))

    print(f"Detected LoRA targets: {unique_targets}")

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=unique_targets,
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # PEFT wraps the model.
    peft_model = get_peft_model(model, config)
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    return peft_model
