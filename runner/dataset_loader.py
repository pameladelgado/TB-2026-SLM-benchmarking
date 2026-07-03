import re

from datasets import load_dataset


# -----------------------------
# Helpers
# -----------------------------

def _load_rows(
    dataset_name: str,
    split: str,
    n: int,
    seed: int | None = None,
    offset: int = 0,
):
    dataset = load_dataset(dataset_name, split=split)

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    start = max(0, offset)
    end = min(start + n, len(dataset))

    if start >= len(dataset):
        return dataset.select([])

    return dataset.select(range(start, end))


def _select_fields(rows, fields: tuple[str, ...]):
    return [{field: row[field] for field in fields} for row in rows]


def _extract_last_number(text):
    if text is None:
        return None

    text = str(text).replace(",", "").replace("$", "").strip()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None

    return matches[-1]


def estimate_token_count(text: str):
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def truncate_text_to_tokens(text: str, max_tokens: int):
    if not text:
        return text

    words = re.findall(r"\S+", text)
    if len(words) <= max_tokens:
        return text

    return " ".join(words[:max_tokens])


def truncate_prompt(prompt: str, max_prompt_tokens: int | None):
    if max_prompt_tokens is None:
        return prompt

    tokens = estimate_token_count(prompt)
    if tokens <= max_prompt_tokens:
        return prompt

    return truncate_text_to_tokens(prompt, max_prompt_tokens)


# -----------------------------
# Benchmark loaders
# -----------------------------

def load_tiny_mmlu(n=20, split="test", seed=None, offset=0):
    rows = _load_rows(
        "tinyBenchmarks/tinyMMLU",
        split,
        n,
        seed=seed,
        offset=offset,
    )
    return _select_fields(rows, ("question", "choices", "answer"))


def load_tiny_hellaswag(n=20, split="validation", seed=None, offset=0):
    rows = _load_rows(
        "tinyBenchmarks/tinyHellaswag",
        split,
        n,
        seed=seed,
        offset=offset,
    )

    samples = []
    for row in rows:
        samples.append(
            {
                "question": row["ctx"],
                "choices": row["endings"],
                "answer": int(row["label"]),
            }
        )

    return samples


def load_tiny_gsm8k(n=20, split="test", seed=None, offset=0):
    rows = _load_rows(
        "tinyBenchmarks/tinyGSM8k",
        split,
        n,
        seed=seed,
        offset=offset,
    )

    samples = []
    for row in rows:
        samples.append(
            {
                "question": row["question"],
                "answer": _extract_last_number(row["answer"]),
            }
        )

    return samples


# -----------------------------
# Few-shot / warm-up loaders
# -----------------------------

def load_tiny_mmlu_few_shot(n=2, seed=None, offset=0):
    return load_tiny_mmlu(n=n, split="dev", seed=seed, offset=offset)


def load_tiny_hellaswag_few_shot(n=2, seed=None, offset=0):
    return load_tiny_hellaswag(n=n, split="train", seed=seed, offset=offset)


def load_tiny_gsm8k_few_shot(n=2, seed=None, offset=0):
    return load_tiny_gsm8k(n=n, split="train", seed=seed, offset=offset)


# -----------------------------
# Dolly workload
# -----------------------------

def load_dolly_workload(n=50, split="train", seed=None, offset=0):
    dataset = load_dataset("databricks/databricks-dolly-15k", split=split)

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    samples = []
    skipped = 0

    for row in dataset:
        if row["category"] in {"closed_qa", "open_qa"}:
            continue

        if skipped < offset:
            skipped += 1
            continue

        samples.append(row)
        if len(samples) >= n:
            break

    return samples


# -----------------------------
# Formatters
# -----------------------------

def format_mcq(sample):
    prompt = sample["question"] + "\n\n"

    for i, choice in enumerate(sample["choices"]):
        prompt += f"{chr(65 + i)}. {choice}\n"

    prompt += "\nChoose the single best answer."
    prompt += "\nReply with exactly one capital letter."
    prompt += "\nDo not explain."
    prompt += "\nAnswer: "

    return prompt


def format_gsm8k(sample):
    return sample["question"] + "\nAnswer: "


def format_dolly(sample, max_prompt_tokens=None):
    instruction = sample["instruction"]
    context = sample.get("context") or ""

    base_prompt = instruction

    if context:
        base_prompt += "\n\nContext:\n" + context

    base_prompt += "\n\nResponse: "

    if max_prompt_tokens is None:
        return base_prompt

    if estimate_token_count(base_prompt) <= max_prompt_tokens:
        return base_prompt

    if context:
        instruction_part = instruction + "\n\nContext:\n"
        suffix = "\n\nResponse: "

        used_tokens = estimate_token_count(instruction_part + suffix)
        remaining_tokens = max_prompt_tokens - used_tokens

        if remaining_tokens > 0:
            truncated_context = truncate_text_to_tokens(context, remaining_tokens)
            return instruction_part + truncated_context + suffix

    return truncate_prompt(base_prompt, max_prompt_tokens)


def format_prompt(sample, task: str, max_prompt_tokens=None):
    if task in {"mmlu", "hellaswag"}:
        prompt = format_mcq(sample)
    elif task == "gsm8k":
        prompt = format_gsm8k(sample)
    elif task == "dolly":
        return format_dolly(sample, max_prompt_tokens=max_prompt_tokens)
    else:
        raise ValueError(f"Unknown task: {task}")

    return truncate_prompt(prompt, max_prompt_tokens)