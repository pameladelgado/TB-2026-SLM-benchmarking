import re

from datasets import load_dataset


# -----------------------------
# Helpers
# -----------------------------

def _load_rows(dataset_name: str, split: str, n: int, seed: int | None = None):
    dataset = load_dataset(dataset_name, split=split)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(min(n, len(dataset))))


def _select_fields(rows, fields: tuple[str, ...]):
    return [{field: row[field] for field in fields} for row in rows]


def _extract_last_number(text):
    if text is None:
        return None

    text = str(text).replace(",", "").replace("$", "").strip()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return text

    return matches[-1]


# -----------------------------
# Benchmark loaders
# -----------------------------

def load_tiny_mmlu(n=20, split="test", seed=None):
    rows = _load_rows("tinyBenchmarks/tinyMMLU", split, n, seed=seed)
    return _select_fields(rows, ("question", "choices", "answer"))


def load_tiny_hellaswag(n=20, split="validation", seed=None):
    rows = _load_rows("tinyBenchmarks/tinyHellaswag", split, n, seed=seed)

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


def load_tiny_gsm8k(n=20, split="test", seed=None):
    rows = _load_rows("tinyBenchmarks/tinyGSM8k", split, n, seed=seed)

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
# Few-shot loaders
# -----------------------------

def load_tiny_mmlu_few_shot(k=2, seed=None):
    return load_tiny_mmlu(n=k, split="dev", seed=seed)


def load_tiny_hellaswag_few_shot(k=2, seed=None):
    return load_tiny_hellaswag(n=k, split="train", seed=seed)


def load_tiny_gsm8k_few_shot(k=2, seed=None):
    return load_tiny_gsm8k(n=k, split="train", seed=seed)


# -----------------------------
# Dolly workload
# -----------------------------

def load_dolly_workload(n=50, split="train", seed=None):
    dataset = load_dataset("databricks/databricks-dolly-15k", split=split)

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    samples = []
    for row in dataset:
        if row["category"] in {"closed_qa", "open_qa"}:
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
    return (
        sample["question"]
        + "\n\nSolve the problem."
        + "\nGive the final numeric answer."
        + "\nDo not explain."
        + "\nAnswer: "
    )


def format_dolly(sample):
    prompt = sample["instruction"]

    if sample.get("context"):
        prompt += "\n\nContext:\n" + sample["context"]

    prompt += "\n\nResponse: "

    return prompt


def format_prompt(sample, task: str):
    if task in {"mmlu", "hellaswag"}:
        return format_mcq(sample)
    if task == "gsm8k":
        return format_gsm8k(sample)
    if task == "dolly":
        return format_dolly(sample)
    raise ValueError(f"Unknown task: {task}")