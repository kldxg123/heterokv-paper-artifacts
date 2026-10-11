"""Original seed/cap helpers extracted without legacy v2 smoke entrypoints."""
import os
import random

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

def set_cap(cap_gib: float):
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, (cap_gib * 1024**3) / float(total))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {"cap_gib": cap_gib, "total_gib": total / 1024**3, "fraction": fraction}
