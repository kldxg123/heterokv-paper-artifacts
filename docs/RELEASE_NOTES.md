# Release notes: 2026-10-11

This is a whitelist-built public snapshot, not a push of the workspace Git history.
It preserves the latest manuscript and experimental numbers. It excludes local
credentials, server/tunnel configuration, SSH orchestration, model weights,
third-party source datasets, internal reviews, temporary renders and archives.
Frozen experimental files are read and hash-verified, never overwritten.

Implementation files retain original bytes except the two generation runners'
import of seed/cap helpers, redirected to an extracted portable helper module.
The architecture builder uses public-relative paths and portable font discovery.
The public release manifest records both original and published source hashes.

Derived results omit raw questions, annotation answers, OCR/summary previews,
absolute model/image paths, server errors and process IDs. Scores remain unchanged;
additional DocVQA ANLS values are computed offline with the paper's >=0.5 rule.
Public identities are dataset sample IDs, not authentication/source credentials.
No new GPU, SSH, VLM, commercial API experiment, dataset download or training is
performed. GitHub publication is the only authenticated external publishing action.
