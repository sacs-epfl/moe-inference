from transformers import SwitchTransformersForConditionalGeneration
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import get_cache, debug
import torch.nn as nn

model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-256", cache_dir=get_cache())

# Assumes no model buffers, checked empirically
def get_model_size(model: nn.Module) -> int:
    """
    Returns the total number of bytes occupied by a PyTorch model (parameters + buffers).
    """
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_size


def bytes_to_GB(size: int) -> float:
    return size / (1024 ** 3)

def get_critical_data_path_model_size(model: nn.Module):
    size_in_bytes = 0

    def count_params(module: nn.Module):
        nonlocal size_in_bytes
        debug(f"Module: {module.__class__.__name__}, Params: {sum(p.numel() for p in module.parameters(recurse=False))}")

        if type(module).__name__ == "SwitchTransformersSparseMLP":
            assert hasattr(module, "experts"), "Module does not have experts"
            one_expert_size = sum(p.numel() * p.element_size() for p in next(iter(module.experts.values())).parameters())
            size_in_bytes += one_expert_size 
            for name, child in module.named_children():
                if name == "experts":
                    continue

                count_params(child)
        else:
            size_in_bytes += sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
            for child in module.children():
                count_params(child)

    count_params(model)
    return size_in_bytes

s = get_model_size(model)
s_dense = get_critical_data_path_model_size(model)
percentage = float(s) / s_dense

print(100 * "-")
print(f"\t- MoE model size: {bytes_to_GB(s):.2f} GiB")
print(f"\t- Dense model size (critical data path): {bytes_to_GB(s_dense):.2f} GiB")
print(f"\t- MoE is {percentage:.2f} times larger than the dense model")
print(100 * "-")

