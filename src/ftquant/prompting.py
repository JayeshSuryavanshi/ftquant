from transformers import PreTrainedTokenizerBase

from ftquant.data import instruction, prompt


def render(tokenizer: PreTrainedTokenizerBase, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def shared_prefix(tokenizer: PreTrainedTokenizerBase, label_set: list[str]) -> list[int]:
    head = render(tokenizer, instruction(label_set).rstrip() + "\x00").split("\x00")[0]
    return encode(tokenizer, head)


def item_tokens(tokenizer: PreTrainedTokenizerBase, message: str, label_set: list[str]) -> list[int]:
    return encode(tokenizer, render(tokenizer, prompt(message, label_set)))
