"""The dataset content-hash recipe: frozen forever, change only deliberately."""

import hashlib
import json
from collections.abc import Sequence

from judgekit.models import CaseRecord

CASE_HASH_FIELDS = ("id", "input", "reference", "human_label", "metadata")


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
