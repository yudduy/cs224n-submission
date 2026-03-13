"""ChatML formatting for Qwen3-8B."""


def format_chatml_prompt(problem: str) -> str:
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"


def format_chatml_response(text: str) -> str:
    return f"{text}<|im_end|>"
