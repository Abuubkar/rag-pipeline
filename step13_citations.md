# Step 13: sources and citations

Notes for [`step13_citations.py`](step13_citations.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Numbered excerpts, model cites [n] | Chunks numbered in the prompt | Simple, readable citations | The model can cite the wrong number |
| Sources list only | Return the retrieved docs, no inline citations | Quick transparency | Doesn't show which claim came from where |
| Quoted evidence | The model must copy an exact quote for each claim | High-trust domains (legal, medical) | Longer answers; quotes still need checking |
| Structured citations | Structured output: answer + list of source ids | UIs that render source links | Guarantees the format, not that the citation is right |
| Post-hoc verification | A second pass checks each claim against its chunk (step 23) | Catching made-up citations | Extra LLM calls |

## This step's code

**When to use**

- Users must be able to check answers.
- You need the retrieved documents next to the answer (debugging, evaluation).

**Limitations**

- The LLM writes the citations: it can cite the wrong number, or none.
- Nothing checks that a claim is really in the cited chunk (grounding, step 23).
- Several chunks of one article look like separate sources ([1]–[4] were all the same CNBC article).
