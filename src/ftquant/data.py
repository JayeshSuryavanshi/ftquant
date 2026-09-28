import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
SPLITS = Path(__file__).resolve().parents[2] / "data" / "banking77"


@dataclass(frozen=True)
class Example:
    text: str
    label: str


def read_split(name: str) -> list[Example]:
    with open(RAW / f"{name}.csv", newline="") as f:
        return [Example(r["text"].strip(), r["category"].strip()) for r in csv.DictReader(f)]


def labels() -> list[str]:
    return sorted({e.label for e in read_split("train")})


def instruction(label_set: list[str]) -> str:
    return (
        "Classify the customer's banking message into exactly one intent. "
        "Answer with the intent name only.\n\nIntents: " + ", ".join(label_set) + "\n\nMessage: "
    )


def prompt(message: str, label_set: list[str]) -> str:
    return instruction(label_set) + message


def stratified_holdout(examples: list[Example], frac: float, seed: int) -> tuple[list[Example], list[Example]]:
    by_label: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_label[e.label].append(e)
    rng = random.Random(seed)
    keep: list[Example] = []
    held: list[Example] = []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n = max(1, round(len(group) * frac))
        held += group[:n]
        keep += group[n:]
    rng.shuffle(keep)
    rng.shuffle(held)
    return keep, held


def write_jsonl(examples: list[Example], path: Path, label_set: list[str]) -> None:
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps({"prompt": prompt(e.text, label_set), "completion": e.label}) + "\n")


def build(seed: int = 0, valid_frac: float = 0.05) -> dict[str, int]:
    label_set = labels()
    train, valid = stratified_holdout(read_split("train"), valid_frac, seed)
    test = read_split("test")
    SPLITS.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, SPLITS / "train.jsonl", label_set)
    write_jsonl(valid, SPLITS / "valid.jsonl", label_set)
    write_jsonl(test, SPLITS / "test.jsonl", label_set)
    return {"train": len(train), "valid": len(valid), "test": len(test), "labels": len(label_set)}


if __name__ == "__main__":
    print(build())
