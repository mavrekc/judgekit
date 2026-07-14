"""The dataset content-hash recipe: frozen forever, change only deliberately."""

import hashlib
import json
from collections.abc import Sequence

from judgekit.models import CaseRecord, JudgeConfigRecord

CASE_HASH_FIELDS = ("id", "input", "reference", "human_label", "metadata")
JUDGE_CONFIG_HASH_FIELDS = ("id", "labels", "model", "params", "rubric")


def case_content_json(record: CaseRecord) -> str:
    """Serialize the hash-relevant fields of a CaseRecord deterministically."""
    content = {
        "id": record.id,
        "input": record.input,
        "reference": record.reference,
        "human_label": record.human_label,
        "metadata": record.metadata,
    }
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dataset_version(records: Sequence[CaseRecord]) -> str:
    """Compute the sha256 content-hash of a dataset's cases, order-independent."""
    ordered = sorted(records, key=lambda record: record.id)
    joined = "\n".join(case_content_json(record) for record in ordered)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def judge_config_content_json(record: JudgeConfigRecord) -> str:
    """Serialize the hash-relevant fields of a JudgeConfigRecord deterministically."""
    content = {
        "id": record.id,
        "labels": list(record.labels),
        "model": record.model,
        "params": {
            "temperature": record.params.temperature,
            "max_tokens": record.params.max_tokens,
            "top_p": record.params.top_p,
            "stop": list(record.params.stop),
        },
        "rubric": record.rubric,
    }
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def judge_config_version_hash(record: JudgeConfigRecord) -> str:
    """Compute the sha256 content-hash of a judge config's behavior fields, frozen forever."""
    digest = hashlib.sha256(judge_config_content_json(record).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
