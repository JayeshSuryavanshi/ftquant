import json
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64}


class SafeTensors:
    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.offset = 8 + n
        self.header.pop("__metadata__", None)
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self) -> list[str]:
        return list(self.header)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.header[name]["shape"])

    def get(self, name: str) -> np.ndarray:
        info = self.header[name]
        start, end = info["data_offsets"]
        raw = self._mm[self.offset + start : self.offset + end]
        dtype = info["dtype"]
        if dtype == "BF16":
            u = raw.view(np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif dtype in _DTYPES:
            arr = raw.view(_DTYPES[dtype]).astype(np.float32)
        else:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        return arr.reshape(info["shape"])


class Checkpoint:
    def __init__(self, folder: Path) -> None:
        files = sorted(folder.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {folder}")
        self.files = [SafeTensors(f) for f in files]
        self.where = {n: st for st in self.files for n in st.names()}

    def names(self) -> list[str]:
        return list(self.where)

    def dtype(self, name: str) -> str:
        return self.where[name].header[name]["dtype"]

    def get(self, name: str) -> np.ndarray:
        return self.where[name].get(name)

    def items(self, names: list[str]) -> Iterator[tuple[str, np.ndarray]]:
        for n in names:
            yield n, self.get(n)


def round_to(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        u = x.astype(np.float32).view(np.uint32)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.view(np.float32)
    if dtype == "F16":
        return x.astype(np.float16).astype(np.float32)
    return x.astype(np.float32)
