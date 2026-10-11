from __future__ import annotations

import json
import os
import subprocess
import sys

from src.memory.document_context_v3 import hashed_multimodal_embedding


def test_hashed_multimodal_embedding_is_normalized() -> None:
    vector = hashed_multimodal_embedding("Invoice total 2026", dim=32)
    assert len(vector) == 32
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-9


def test_hashed_multimodal_embedding_is_process_stable() -> None:
    code = (
        "import json; "
        "from src.memory.document_context_v3 import hashed_multimodal_embedding; "
        "print(json.dumps(hashed_multimodal_embedding('Invoice total 2026', dim=32)))"
    )
    outputs = []
    for seed in ("1", "777"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                text=True,
                env=environment,
            ).strip()
        )
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0]) == hashed_multimodal_embedding("Invoice total 2026", dim=32)
