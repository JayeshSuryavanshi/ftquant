import json
import re
from dataclasses import dataclass
from pathlib import Path

from ftquant import data as banking77

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
SPLITS = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Item:
    text: str
    target: str


class Task:
    name: str
    max_tokens: int

    def split(self, name: str) -> list[Item]:
        raise NotImplementedError

    def prompt(self, text: str) -> str:
        raise NotImplementedError

    def prefix(self) -> str:
        raise NotImplementedError

    def correct(self, pred: str, target: str) -> bool:
        return pred.strip() == target.strip()

    def valid(self, pred: str) -> bool:
        return bool(pred.strip())


class Banking77(Task):
    name = "banking77"
    max_tokens = 16

    def __init__(self) -> None:
        self.labels = banking77.labels()

    def split(self, name: str) -> list[Item]:
        return [Item(e.text, e.label) for e in banking77.read_split(name)]

    def prefix(self) -> str:
        return banking77.instruction(self.labels)

    def prompt(self, text: str) -> str:
        return banking77.prompt(text, self.labels)

    def valid(self, pred: str) -> bool:
        return pred.strip() in self.labels


class Massive(Task):
    name = "massive"
    max_tokens = 64
    SLOT = re.compile(r"\[(\w+) :")

    def __init__(self) -> None:
        rows = [json.loads(line) for line in open(RAW / "massive" / "en-US.jsonl")]
        self.rows = rows
        self.slots = sorted(
            {m for r in rows for m in self.SLOT.findall(r["annot_utt"])}
        )

    def split(self, name: str) -> list[Item]:
        part = {"train": "train", "valid": "dev", "test": "test"}[name]
        return [
            Item(r["utt"], r["annot_utt"]) for r in self.rows if r["partition"] == part
        ]

    def prefix(self) -> str:
        return (
            "Rewrite the user's request, marking every slot as [slot_type : words]. "
            "Copy all other words unchanged. Answer with the rewritten request only.\n\n"
            "Slot types: " + ", ".join(self.slots) + "\n\nRequest: "
        )

    def prompt(self, text: str) -> str:
        return self.prefix() + text

    def correct(self, pred: str, target: str) -> bool:
        return " ".join(pred.split()) == " ".join(target.split())


def get(name: str) -> Task:
    return {"banking77": Banking77, "massive": Massive}[name]()


def write_training_files(task: Task, seed: int = 0, valid_frac: float = 0.05) -> Path:
    import random

    if task.name == "banking77":
        raise ValueError("banking77 training files come from ftquant.data.build (stratified split); do not overwrite them")

    out = SPLITS / task.name
    out.mkdir(parents=True, exist_ok=True)
    train = task.split("train")
    rng = random.Random(seed)
    rng.shuffle(train)
    if task.name == "massive":
        valid = task.split("valid")
    else:
        n = max(1, int(len(train) * valid_frac))
        valid, train = train[:n], train[n:]
    for fname, items in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(out / fname, "w") as f:
            for it in items:
                f.write(
                    json.dumps(
                        {"prompt": task.prompt(it.text), "completion": it.target}
                    )
                    + "\n"
                )
    return out
