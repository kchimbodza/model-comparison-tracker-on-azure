"""Run one question set against several models and log each run to MLflow.

Usage:
    uv run python compare.py azure:gpt-4.1-mini ollama:qwen2.5:7b

Each model gets one MLflow run with:
    params    provider, model, prompt version, test set version (+ file hashes)
    metrics   accuracy, average latency, total tokens, cost in USD
    artifacts the prompt, the test set, and a table of every answer
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import mlflow

ROOT = Path(__file__).parent
PROMPT_FILE = ROOT / "prompts" / "system_v1.txt"
TEST_FILE = ROOT / "data" / "questions_v1.jsonl"
EXPERIMENT = "model-comparison"


# ---------------------------------------------------------------- setup

def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env into the environment (keeps the key out of the code)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def sha256(path: Path) -> str:
    """Short fingerprint of a file, so every run records exactly which prompt and test set it used."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def version_of(path: Path) -> str:
    """'questions_v1.jsonl' -> 'v1'."""
    match = re.search(r"_(v\d+)\.", path.name)
    return match.group(1) if match else "unknown"


# ---------------------------------------------------------------- scoring

def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def is_correct(answer: str, accept: list[str]) -> bool:
    """Correct if any accepted answer appears as a whole word (or phrase) in the model's answer."""
    words = f" {normalise(answer)} "
    return any(f" {normalise(a)} " in words for a in accept)


# ---------------------------------------------------------------- models

def ask_azure(deployment: str, system: str, question: str) -> tuple[str, int, int, float]:
    """Returns (answer, tokens in, tokens out, seconds spent waiting on rate limits)."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    waited = 0.0
    for attempt in range(5):
        response = httpx.post(
            f"{endpoint}/openai/deployments/{deployment}/chat/completions",
            params={"api-version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")},
            headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]},
            json={"messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
                  "max_tokens": 100, "temperature": 0, "response_format": {"type": "json_object"}},
            timeout=60,
        )
        if response.status_code != 429:        # 429 = rate limit: wait as long as Azure asks, then retry
            break
        pause = float(response.headers.get("retry-after", 10))
        print(f"      rate limited, waiting {pause:.0f} s")
        time.sleep(pause)
        waited += pause
    response.raise_for_status()
    data = response.json()
    answer = json.loads(data["choices"][0]["message"]["content"]).get("answer", "")
    return answer, data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"], waited


def ask_ollama(model: str, system: str, question: str) -> tuple[str, int, int, float]:
    response = httpx.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}]},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    answer = json.loads(data["message"]["content"]).get("answer", "")
    return answer, data.get("prompt_eval_count", 0), data.get("eval_count", 0), 0.0


def cost_usd(provider: str, tokens_in: int, tokens_out: int) -> float:
    """Local models are free to run. Azure is priced per million tokens (rates come from .env)."""
    if provider != "azure":
        return 0.0
    rate_in = float(os.environ.get("AZURE_PRICE_IN", 0))
    rate_out = float(os.environ.get("AZURE_PRICE_OUT", 0))
    return (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000


# ---------------------------------------------------------------- one run

def run_model(spec: str, system: str, questions: list[dict]) -> None:
    provider, model = spec.split(":", 1)
    ask = {"azure": ask_azure, "ollama": ask_ollama}[provider]
    rows, retries = [], 0
    print(f"\n{spec}")
    for q in questions:
        start = time.perf_counter()
        try:
            answer, tokens_in, tokens_out, waited = ask(model, system, q["question"])
        except Exception as e:                      # a failed call counts as a wrong answer
            answer, tokens_in, tokens_out, waited = f"ERROR: {type(e).__name__}", 0, 0, 0.0
        latency_ms = (time.perf_counter() - start - waited) * 1000   # rate-limit waits are not model speed
        retries += waited > 0
        correct = is_correct(answer, q["accept"])
        rows.append({"id": q["id"], "question": q["question"], "answer": answer, "correct": correct,
                     "latency_ms": round(latency_ms), "tokens_in": tokens_in, "tokens_out": tokens_out})
        print(f"  {'✓' if correct else '✗'} Q{q['id']:>2}  {latency_ms:6.0f} ms  {answer}")

    tokens_in = sum(r["tokens_in"] for r in rows)
    tokens_out = sum(r["tokens_out"] for r in rows)
    metrics = {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
        "tokens_total": tokens_in + tokens_out,
        "cost_usd": cost_usd(provider, tokens_in, tokens_out),
        "rate_limit_retries": retries,
    }

    with mlflow.start_run(run_name=spec):
        mlflow.log_params({
            "provider": provider, "model": model,
            "prompt_version": version_of(PROMPT_FILE), "prompt_sha": sha256(PROMPT_FILE),
            "test_set_version": version_of(TEST_FILE), "test_set_sha": sha256(TEST_FILE),
            "questions": len(rows),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(PROMPT_FILE), "inputs")
        mlflow.log_artifact(str(TEST_FILE), "inputs")
        mlflow.log_table({k: [r[k] for r in rows] for k in rows[0]}, "answers.json")

    print(f"  accuracy {metrics['accuracy']:.0%} · avg {metrics['avg_latency_ms']:.0f} ms · "
          f"{metrics['tokens_total']} tokens · ${metrics['cost_usd']:.6f}")


def main() -> None:
    specs = sys.argv[1:] or ["azure:gpt-4.1-mini", "ollama:qwen2.5:7b"]
    load_env()
    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    system = PROMPT_FILE.read_text()
    questions = [json.loads(line) for line in TEST_FILE.read_text().splitlines() if line.strip()]
    for spec in specs:
        run_model(spec, system, questions)
    print("\nOpen the comparison: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
