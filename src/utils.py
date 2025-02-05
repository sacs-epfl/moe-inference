from enum import Enum
import torch
import torch.nn as nn
import torch.distributed as dist
import os
from constants import DEFAULT_CACHE_DIR, DEBUG

def str2bool(s):
    return s.lower() in ["yes", "y", "true", "t"]

def get_cache():
    if "CACHE" in os.environ:
        return os.environ["CACHE"]
    else:
        print("Cache directory not set, using default ", DEFAULT_CACHE_DIR)
        return DEFAULT_CACHE_DIR

def debug(text, rank=0):
    if DEBUG and dist.get_rank() == rank:
        print(f"[DEBUG][{rank}] - ",text)

def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x

class ChunkType(Enum):
    """
    Enum for the type of chunking to use in the model.
    """

    ROW = "row"
    COLUMN = "column"


class TimedModule(nn.Module):
    def __init__(self, child, idx=0, is_decoder=False):
        super().__init__()

        self.child = child
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.latencies = []
        self.idx = idx 
        self.is_decoder = is_decoder
    
    def forward(self, x):
        self.start.record()
        x = self.child(x)
        self.end.record()
        self.end.synchronize()
        self.latencies.append(self.start.elapsed_time(self.end))
        return x
    
    def get_latencies(self):
        return self.latencies[:]
    

def get_timing_modules(acc, module):
    if type(module).__name__ == "TimedModule":
        acc.append(module)
    else:
        for child in module.children():
            acc = get_timing_modules(acc, child)
    return acc 