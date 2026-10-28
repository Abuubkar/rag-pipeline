# Step 18: metadata filtering ("only TechCrunch articles")

Notes for [`step18_metadata_filter.py`](step18_metadata_filter.py). Before/after results: [`step18_comparison.md`](step18_comparison.md).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ LLM extracts filters ("self-query") | Structured output turns the question into (query, filter) pairs | Questions that name a publication, author, category or date | The LLM can miss a filter or pick the wrong value; a too-strict filter returns nothing |
| SelfQueryRetriever (langchain-classic) | Ready-made: the LLM writes a structured query, a translator turns it into the store's filter | Quick setup on a supported store | Legacy package |
| Filters chosen by the user | Dropdowns or facets in the UI (source, date range) | Apps where users know what they want | Users must know the values; no help for free-text questions |
| ✅ Filter inside the search (pre-filtering) | `WHERE source = …`, then the vector search on what's left | You need exact restriction | With approximate indexes (HNSW), a strict filter can return fewer than k results |
| Post-filtering | Search the top N first, then drop non-matching chunks | Stores without filter support | Can end up with too few results |
| Metadata in the query text (step 16) | Put "Sporting News" in the query | Nothing to set up | Only nudges the vector; other publications still come back |
| One collection per group | A separate table or index per source or customer | A few fixed groups (e.g. one per tenant) | Many collections to manage; cross-group questions need several searches |

## This step's code

**When to use**

- Questions that say where something was reported ("as reported by CBSSports.com and Sporting News").
- Metadata has a small, known set of values (49 publications) that can be listed in the prompt.

**Limitations**

- Only `source` is filtered. Dates ("published at 13:41:30", "December 6, 2023") aren't yet.
- The LLM only adds a filter when the question ties a publication to that part: "NBA betting promotions" stayed
  unfiltered, so fact B (Sporting News, "NBA Rookie of the Year Odds") was still missed.
- `Literal[...]` forces an exact name, but not the right one: the model can still pick a wrong publication.
- All 49 names go into every prompt. Thousands of values (authors, product ids) would need another approach,
  such as matching the name against the list in code.
- More searches mean more chunks (22 for the betting question, up from 5), so reranking (step 17) matters more.
