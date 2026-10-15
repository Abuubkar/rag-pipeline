# 0008. Prompts live in git and carry a version

- Status: accepted
- Date: 2026-10-16

## Context
Results are only reproducible if we know exactly which prompt produced them.

## Decision
Prompts are YAML files in `backend/config/prompts/` with `name` and `version`. Every answer records
`name@version` (e.g. `answer@1`). Bump `version` on any change. Langfuse prompt management will mirror
these in M4 (`main.py prompts push`); git stays the source of truth.

## Consequences
Prompt changes show up in code review and git history. Langfuse is a mirror, not where prompts are edited.
