"""
Langfuse integration for rag_langfuse_project: container connectivity, LangGraph
tracing, analytics-dashboard aggregation, and LLM-as-judge evaluation.

Scoped to a standalone, single-user setup — no external database or service
beyond the Langfuse container itself. Everything here is a no-op when LANGFUSE_ENABLED is not "true", so the RAG
pipeline in rag_graph.py works with or without the container running.
"""

import contextlib
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

PROJECT_ID = "rag-langfuse-project"
_langfuse_client = None


def _is_enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "false").lower() == "true" and bool(os.getenv("LANGFUSE_HOST"))


def get_langfuse_client():
    """Singleton Langfuse SDK client for querying traces/observations and scoring."""
    if not _is_enabled():
        return None
    global _langfuse_client
    if _langfuse_client is None:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )
    return _langfuse_client


def verify_langfuse_connection() -> bool:
    """Check that the self-hosted Langfuse container (docker-compose.langfuse.yml) is reachable."""
    if not _is_enabled():
        logger.warning("[langfuse] Disabled — set LANGFUSE_ENABLED=true and LANGFUSE_HOST in .env")
        return False
    try:
        import requests

        host = os.getenv("LANGFUSE_HOST", "").rstrip("/")
        resp = requests.get(f"{host}/api/public/health", timeout=10)
        ok = resp.status_code == 200
        logger.info(f"[langfuse] {'Reachable' if ok else 'Unreachable'} at {host} (HTTP {resp.status_code})")
        return ok
    except Exception as e:
        logger.warning(f"[langfuse] Connection check failed: {e}")
        return False


@contextlib.contextmanager
def traced_run(
    trace_name: str,
    session_id: Optional[str] = None,
    user_id: str = "rag_langfuse_project",
    tags: Optional[List[str]] = None,
):
    """
    Set Langfuse trace attributes and a LangChain/LangGraph CallbackHandler for
    one graph run. Yields ``(config, trace_id)`` — pass ``config`` straight into
    ``graph.invoke(state, config=config)``; ``trace_id`` lets you attach an
    evaluation score to this exact trace afterwards via ``evaluate_answer()``.

    Uses the Langfuse SDK v4 ``propagate_attributes()`` + ``CallbackHandler
    (trace_context=...)`` pattern. A no-op when Langfuse is
    disabled, so callers don't need to branch on ``_is_enabled()`` themselves.
    """
    trace_id = uuid.uuid4().hex
    config: Dict[str, Any] = {}

    if not _is_enabled():
        yield config, trace_id
        return

    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    with propagate_attributes(
        user_id=user_id,
        session_id=session_id or trace_id,
        tags=tags or ["rag_langfuse_project"],
        trace_name=trace_name,
    ):
        handler = CallbackHandler(trace_context={"trace_id": trace_id, "user_id": user_id})
        config["callbacks"] = [handler]
        yield config, trace_id


def flush_langfuse() -> None:
    """Flush buffered events so a short-lived CLI process doesn't drop traces on exit."""
    try:
        client = get_langfuse_client()
        if client:
            client.flush()
    except Exception as e:
        logger.warning(f"[langfuse] Flush failed: {e}")


# ── Analytics dashboard ──────────────────────────────────────────────────
#
# NOTE: `Trace` objects have no per-trace token usage field in the installed
# langfuse SDK (only `total_cost` and `latency`) — token counts live on
# `Observation.usage_details` (generation spans), so token/cost-per-model
# stats are aggregated from observations, not traces.

def _public_host() -> str:
    return os.getenv("LANGFUSE_HOST", "http://localhost:3100").rstrip("/")


def _fetch_traces(cutoff: datetime, max_pages: int = 10) -> List[Any]:
    client = get_langfuse_client()
    traces: List[Any] = []
    page = 1
    while page <= max_pages:
        resp = client.api.trace.list(limit=100, page=page, from_timestamp=cutoff)
        batch = resp.data if hasattr(resp, "data") else []
        if not batch:
            break
        traces.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return traces


def _fetch_observations(cutoff: Optional[datetime] = None, max_pages: int = 10) -> List[Any]:
    """Fetch observations via the legacy v1 endpoint (page-based, and exposes
    the deprecated-but-populated `usage`/`model` fields)."""
    client = get_langfuse_client()
    observations: List[Any] = []
    page = 1
    while page <= max_pages:
        resp = client.api.legacy.observations_v1.get_many(limit=100, page=page, from_start_time=cutoff)
        batch = resp.data if hasattr(resp, "data") else []
        if not batch:
            break
        observations.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return observations


def get_dashboard_summary(days: int = 30) -> Dict[str, Any]:
    """Aggregate total traces, tokens, cost, and avg latency over the last N days."""
    client = get_langfuse_client()
    if not client:
        return {"status": "disabled"}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    traces = _fetch_traces(cutoff)
    observations = _fetch_observations(cutoff)

    total_cost = sum(getattr(t, "total_cost", 0) or 0 for t in traces)
    latencies = [t.latency for t in traces if getattr(t, "latency", None)]
    input_tokens = sum(getattr(o, "usage", None) and o.usage.input or 0 for o in observations)
    output_tokens = sum(getattr(o, "usage", None) and o.usage.output or 0 for o in observations)

    return {
        "dashboard_url": f"{_public_host()}/project/{PROJECT_ID}",
        "total_traces": len(traces),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": int(sum(latencies) / len(latencies) * 1000) if latencies else 0,
    }


def get_daily_usage_trends(days: int = 30) -> Dict[str, Any]:
    """Daily trace-count / token / cost trend, ready for charting."""
    client = get_langfuse_client()
    if not client:
        return {"status": "disabled"}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    traces = _fetch_traces(cutoff)
    observations = _fetch_observations(cutoff)

    daily = defaultdict(lambda: {"traces": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    for t in traces:
        if not t.timestamp:
            continue
        daily[t.timestamp.date().isoformat()]["traces"] += 1

    for o in observations:
        if not o.start_time:
            continue
        key = o.start_time.date().isoformat()
        if getattr(o, "usage", None):
            daily[key]["input_tokens"] += o.usage.input or 0
            daily[key]["output_tokens"] += o.usage.output or 0
            daily[key]["cost_usd"] += o.usage.total_cost or 0

    trend = [{"date": d, **v} for d, v in sorted(daily.items())]
    return {"days": days, "trend": trend}


def get_model_cost_breakdown(days: int = 30) -> Dict[str, Any]:
    """Cost / token breakdown per LLM model, aggregated from observations."""
    client = get_langfuse_client()
    if not client:
        return {"status": "disabled"}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    observations = _fetch_observations(cutoff)

    per_model = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    for obs in observations:
        if not obs.model:
            continue
        per_model[obs.model]["calls"] += 1
        if getattr(obs, "usage", None):
            per_model[obs.model]["input_tokens"] += obs.usage.input or 0
            per_model[obs.model]["output_tokens"] += obs.usage.output or 0
            per_model[obs.model]["cost_usd"] += obs.usage.total_cost or 0

    breakdown = sorted(
        ({"model": m, **v} for m, v in per_model.items()),
        key=lambda x: x["cost_usd"],
        reverse=True,
    )
    return {"models": breakdown}


# ── LLM-as-judge evaluation ──────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a PDF-RAG assistant's answer.

Question: {question}

Retrieved context:
{context}

Assistant answer: {answer}

Score each 0-100 and respond with JSON only, no markdown fences:
{{
  "faithfulness": <0-100, is the answer supported by the context>,
  "relevance": <0-100, does it address the question>,
  "completeness": <0-100, does it cover the context that matters>,
  "overall": <0-100>,
  "comment": "one sentence explanation"
}}"""


def _parse_json_response(text: str) -> Dict[str, Any]:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LLM-as-judge scoring of one RAG answer (faithfulness / relevance /
    completeness / overall). If Langfuse is enabled and a ``trace_id`` is
    given (the one returned by ``traced_run``), the "overall" score is pushed
    onto that trace so it shows up in the Langfuse UI's Scores tab.
    """
    from langchain_openai import ChatOpenAI

    judge = ChatOpenAI(model=os.getenv("CHAT_MODEL", "openai/gpt-oss-120b"), temperature=0)
    prompt = JUDGE_PROMPT.format(question=question, context=context[:4000], answer=answer[:2000])

    try:
        response = judge.invoke(prompt)
        result = _parse_json_response(response.content)
    except Exception as e:
        logger.error(f"[eval] Judge call failed: {e}")
        return {"overall": 0, "error": str(e)}

    client = get_langfuse_client()
    if client and trace_id:
        try:
            client.create_score(
                trace_id=trace_id,
                name="overall",
                value=result.get("overall", 0) / 100.0,
                data_type="NUMERIC",
                comment=result.get("comment"),
            )
            client.flush()
        except Exception as e:
            logger.warning(f"[eval] Failed to push score to Langfuse: {e}")

    logger.info(
        f"[eval] overall={result.get('overall')} faithfulness={result.get('faithfulness')} "
        f"relevance={result.get('relevance')} completeness={result.get('completeness')}"
    )
    return result


# ── Golden-dataset evaluation (Langfuse Datasets + Experiments) ──────────
#
# Feeds a rag_eval_questions.json-style file (question/answer/key_facts
# pairs) into a Langfuse Dataset, then runs every item through the RAG graph
# as a Langfuse Experiment. Each item's expected answer is attached to its
# span as the experiment's "expected output", so a Langfuse-managed
# evaluator (e.g. the Ragas "Answer Correctness" template) configured with
# "Run on: Experiments" can map ground_truth -> Expected Output and
# answer -> Output directly — no manual metadata wiring needed.

def load_eval_dataset(json_path: str, dataset_name: Optional[str] = None) -> Optional[str]:
    """Upload a rag_eval_questions.json-style file into a Langfuse Dataset. Idempotent — re-running skips questions already present (matched by their `id`)."""
    client = get_langfuse_client()
    if not client:
        logger.warning("[eval] Langfuse disabled — cannot create dataset.")
        return None

    with open(json_path) as f:
        data = json.load(f)

    dataset_name = dataset_name or os.path.splitext(data.get("corpus", "rag-eval-questions"))[0]

    try:
        dataset = client.get_dataset(dataset_name)
        existing_ids = {
            item.metadata.get("id") for item in dataset.items if isinstance(item.metadata, dict)
        }
    except Exception:
        client.create_dataset(name=dataset_name, description=data.get("corpus_note"))
        existing_ids = set()

    created = 0
    for q in data["questions"]:
        if q.get("id") in existing_ids:
            continue
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": q["question"]},
            expected_output=q["answer"],
            metadata={k: v for k, v in q.items() if k not in ("question", "answer")},
        )
        created += 1

    logger.info(f"[eval] Dataset '{dataset_name}': added {created} new item(s).")
    return dataset_name


def run_eval_experiment(dataset_name: str, run_name: Optional[str] = None) -> Dict[str, Any]:
    """Run every item in a Langfuse dataset through the RAG graph as one Experiment run."""
    client = get_langfuse_client()
    if not client:
        logger.warning("[eval] Langfuse disabled — cannot run experiment.")
        return {"status": "disabled"}

    from langfuse.langchain import CallbackHandler

    from rag_graph import build_graph

    app = build_graph()

    def task(*, item, **kwargs):
        # No traced_run() here: run_experiment() already opens the trace/span
        # for this item and propagates it via context, so a bare
        # CallbackHandler() nests under it automatically instead of starting
        # a second, unlinked trace.
        state = {"question": item.input["question"], "context": [], "answer": ""}
        result = app.invoke(state, config={"callbacks": [CallbackHandler()]})
        return result["answer"]

    dataset = client.get_dataset(dataset_name)
    result = dataset.run_experiment(name=dataset_name, run_name=run_name, task=task)
    flush_langfuse()

    logger.info(f"[eval] Experiment '{result.run_name}' finished — {len(result.item_results)} item(s).")
    return {
        "run_name": result.run_name,
        "items": len(result.item_results),
        "dataset_run_url": getattr(result, "dataset_run_url", None),
    }


if __name__ == "__main__":
    import sys

    command, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("verify", [])
    if command == "verify":
        verify_langfuse_connection()
    elif command == "dashboard":
        print(json.dumps(get_dashboard_summary(), indent=2, default=str))
    elif command == "trends":
        print(json.dumps(get_daily_usage_trends(), indent=2, default=str))
    elif command == "models":
        print(json.dumps(get_model_cost_breakdown(), indent=2, default=str))
    elif command == "load-dataset":
        if not args:
            print("Usage: python observability.py load-dataset <path/to/questions.json> [dataset_name]")
            sys.exit(1)
        name = load_eval_dataset(args[0], args[1] if len(args) > 1 else None)
        print(f"Dataset ready: {name}")
    elif command == "run-experiment":
        if not args:
            print("Usage: python observability.py run-experiment <dataset_name> [run_name]")
            sys.exit(1)
        print(json.dumps(run_eval_experiment(args[0], args[1] if len(args) > 1 else None), indent=2, default=str))
    else:
        print(
            "Usage: python observability.py "
            "[verify|dashboard|trends|models|load-dataset <json> [name]|run-experiment <dataset> [run_name]]"
        )
