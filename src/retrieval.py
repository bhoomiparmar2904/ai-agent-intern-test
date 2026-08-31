"""
Retrieval over the knowledge-base markdown files.

Design choice: TF-IDF (scikit-learn) cosine similarity instead of a hosted embedding
API. This keeps retrieval fully local, deterministic, dependency-light, and testable
without any network calls or extra API keys -- important for a reproducible eval suite.
See README "Architecture" section for the tradeoffs of this choice.
"""
from __future__ import annotations

import re
import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc_id: str          # filename, e.g. "01-returns-policy-current.md"
    heading: str         # nearest markdown heading, e.g. "Standard return window"
    text: str            # chunk body text
    status: str = "active"   # from front matter: active | superseded | internal
    order: int = 0            # position in file, used as a stable tiebreaker

    @property
    def source(self) -> str:
        return f"{self.doc_id} — {self.heading}" if self.heading else self.doc_id


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split YAML front matter (--- ... ---) from the markdown body."""
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            body = parts[2]
            return meta, body
    return {}, raw


def _chunk_markdown(doc_id: str, meta: dict, body: str) -> list[Chunk]:
    """Split a markdown body into chunks on '## ' headings."""
    status = meta.get("status", "active")
    chunks: list[Chunk] = []
    # Split on level-2 headings, keeping the heading with its section.
    sections = re.split(r"(?m)^##\s+", body)
    order = 0
    # First section before any "## " heading (may include a title "# ...")
    intro = sections[0].strip()
    if intro:
        title_match = re.match(r"^#\s+(.+)$", intro, re.MULTILINE)
        heading = title_match.group(1).strip() if title_match else meta.get("title", doc_id)
        text = re.sub(r"^#\s+.+$", "", intro, count=1, flags=re.MULTILINE).strip()
        if text:
            chunks.append(Chunk(doc_id=doc_id, heading=heading, text=text, status=status, order=order))
            order += 1
    for section in sections[1:]:
        lines = section.strip().split("\n", 1)
        heading = lines[0].strip()
        text = lines[1].strip() if len(lines) > 1 else ""
        if text:
            chunks.append(Chunk(doc_id=doc_id, heading=heading, text=text, status=status, order=order))
            order += 1
    return chunks


class Retriever:
    def __init__(self, kb_dir: str):
        self.kb_dir = kb_dir
        self.chunks: list[Chunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._load()

    def _load(self):
        for path in sorted(glob.glob(os.path.join(self.kb_dir, "*.md"))):
            doc_id = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            meta, body = _parse_front_matter(raw)
            self.chunks.extend(_chunk_markdown(doc_id, meta, body))

        if not self.chunks:
            raise RuntimeError(f"No knowledge-base chunks found in {self.kb_dir}")

        corpus = [f"{c.doc_id} {c.heading} {c.text}" for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k chunks as dicts with similarity score, ranked with a
        preference for active (non-superseded, non-internal) documents."""
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        scored = []
        for chunk, score in zip(self.chunks, sims):
            if score <= 0:
                continue
            # Document-precedence boost: prefer active docs over superseded/internal ones.
            boost = {"active": 1.0, "superseded": 0.55, "internal": 0.4}.get(chunk.status, 0.8)
            scored.append((score * boost, score, chunk))

        scored.sort(key=lambda t: (-t[0], t[2].order))
        results = []
        for boosted_score, raw_score, chunk in scored[:top_k]:
            results.append({
                "doc_id": chunk.doc_id,
                "heading": chunk.heading,
                "source": chunk.source,
                "status": chunk.status,
                "text": chunk.text,
                "score": round(float(raw_score), 4),
            })
        return results
