import json
import struct
from pathlib import Path

import numpy as np


def to_bf16_bytes(x: np.ndarray) -> bytes:
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16).tobytes()


def write_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
    header, blobs, offset = {}, [], 0
    for name, (arr, dtype) in tensors.items():
        data = to_bf16_bytes(arr) if dtype == "BF16" else arr.astype(np.float32).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
