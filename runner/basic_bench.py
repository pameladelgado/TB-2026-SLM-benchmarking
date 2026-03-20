import json
import time
import requests

URL = "http://localhost:8080/completion"

payload = {
    "prompt": "Answer in one word: capital of France?",
    "n_predict": 20,
    "temperature": 0,
    "stream": True,
}

start = time.perf_counter()
ttft = None
chunks = []

with requests.post(URL, json=payload, stream=True) as r:
    r.raise_for_status()

    for raw_line in r.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        line = raw_line.strip()

        if line.startswith("data:"):
            line = line[len("data:"):].strip()

        if line == "[DONE]":
            break

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print("Skipping non-JSON line:", repr(raw_line))
            continue

        content = data.get("content", "")

        if content and ttft is None:
            ttft = time.perf_counter() - start

        if content:
            chunks.append(content)

        if data.get("stop", False):
            break

total_latency = time.perf_counter() - start
output_text = "".join(chunks)

print("TTFT:", ttft)
print("Total latency:", total_latency)
print("Output:", output_text)
