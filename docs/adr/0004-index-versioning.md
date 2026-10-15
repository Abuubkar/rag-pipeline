# 0004. Content-addressed index versions

- Status: accepted
- Date: 2026-10-16

## Context
A vector is only meaningful next to the model, prefixes, chunker and cleaning rules that produced it.
Naming indexes by chunking strategy alone allows silent mixing of incompatible vectors.

## Decision
- An **index manifest** records dataset (name, revision, content hash), preprocessing
  (cleaning version), chunking (library, splitter, size, overlap, header, library version) and the
  dense embedder (provider, model, Ollama digest, dims, document/query prefixes, normalisation).
- `version_id = "{splitter}-{sha256(canonical manifest)[:10]}"`.
- Chunk id = `sha256(version|document|ordinal)` → inserts are idempotent; a crashed run resumes.
- Status: `building → ready | failed` (error stored). Ready means stored count == expected count.
- `active_index(alias → version)` decides what `ask` queries; `main.py index activate` switches it.
- **Query guard:** the container builds the query embedder from the active manifest and raises
  `EmbedderMismatch` if config or the local Ollama model digest differ.

## Consequences
Re-running `ingest` with unchanged inputs is a no-op. Any experiment (chunk size, header on/off) creates a
new version side by side, which makes A/B comparison in Milestone 2/3 trivial. Old versions stay until
they're retired (cleanup command is backlog).
