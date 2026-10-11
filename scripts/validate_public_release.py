"""CPU-only checksum, disclosure and dependency audit of the public release."""
from __future__ import annotations
import ast
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    errors = []
    patterns = {
        "credential": r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{16,})",
        "private_key": r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
        "private_user_path": r"(?:[A-Z]:[\\/]Users[\\/][^\s]+|/root/[^\s]+|/home/[^\s]+)",
        "remote_ssh": r"(?:workstation-vm201-remote|IdentityFile|ProxyJump|sshpass)",
        "unexpected_ip": r"(?<![\d.])(?!(?:127\.0\.0\.1|0\.0\.0\.0)\b)(?:\d{1,3}\.){3}\d{1,3}(?![\d.])",
    }
    hash_checked, python_parsed, derived_runs = 0, 0, 0
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file():
            errors.append({"path": item["path"], "reason": "missing/unsafe path"})
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            errors.append({"path": item["path"], "reason": "checksum mismatch"})
        hash_checked += 1
        if len(data) >= 90 * 1024 * 1024:
            errors.append({"path": item["path"], "reason": "too large for GitHub"})
        text = None
        if path.suffix == ".gz":
            text = gzip.decompress(data).decode("utf-8")
            record = json.loads(text)
            if record.get("sanitized_derived_records"):
                derived_runs += 1
                for row in record["rows"] + record.get("warmup_rows", []):
                    if any(key in row for key in ("question", "answers", "raw_pred", "target_searchable_text_preview", "selected_image_path")):
                        errors.append({"path": item["path"], "reason": "forbidden raw dataset field"})
        elif path.suffix in (".py", ".md", ".json", ".jsonl", ".csv", ".tex", ".bib", ".svg", ".drawio", ".txt", ".ini", ".cff"):
            text = data.decode("utf-8")
        if path.suffix == ".py":
            ast.parse(text.lstrip("\ufeff"), filename=item["path"])
            python_parsed += 1
        # The validator contains regex examples, not actual credentials/hosts.
        if text is not None and path != Path(__file__):
            for name, pattern in patterns.items():
                if re.search(pattern, text, re.I):
                    errors.append({"path": item["path"], "reason": name})
    if derived_runs != 68:
        errors.append({"reason": "expected 68 current E1-E4 derived runs", "actual": derived_runs})
    from pypdf import PdfReader
    paper = PdfReader(ROOT / "paper/main.pdf")
    if len(paper.pages) != 6:
        errors.append({"reason": "unexpected paper page count"})
    paper_text = "\n".join(page.extract_text() or "" for page in paper.pages)
    for name in ("credential", "private_user_path", "remote_ssh", "unexpected_ip"):
        if re.search(patterns[name], paper_text, re.I):
            errors.append({"path": "paper/main.pdf", "reason": name})
    result = {"status": "passed" if not errors else "failed", "files_hash_checked": hash_checked,
              "python_files_parsed": python_parsed, "derived_runs": derived_runs, "paper_pages": len(paper.pages),
              "secret_findings": errors, "gpu_jobs": 0, "network_calls": 0}
    (ROOT / "PUBLIC_VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "passed" else 1)
