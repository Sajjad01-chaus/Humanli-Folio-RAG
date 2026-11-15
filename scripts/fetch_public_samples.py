"""Download unmodified public technical reports from NIST's official publication server."""

import hashlib
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    (
        "nist-ai-rmf-1.0.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
        "2023-01",
    ),
    (
        "nist-genai-profile.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile",
        "2024-07",
    ),
]


def main() -> None:
    folder = ROOT / "samples" / "public"
    folder.mkdir(exist_ok=True)
    manifest = []
    for filename, url, title, date in REPORTS:
        response = httpx.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF-"):
            raise ValueError("The official server did not return a PDF.")
        (folder / filename).write_bytes(response.content)
        manifest.append(
            {
                "filename": filename,
                "title": title,
                "publication_date": date,
                "source": url,
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "bytes": len(response.content),
            }
        )
        print(filename, len(response.content), "bytes")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
