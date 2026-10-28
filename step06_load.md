# Step 6: load documents

Notes for [`step06_load.py`](step06_load.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Custom BaseLoader | Write lazy_load() yourself | No maintained loader exists; you need control (revision pinning) | You maintain it |
| Ready-made loaders (langchain-community) | Hundreds of sources: web, Notion, S3… | Quick prototypes | The package is sunset; quality varies (the old HF loader corrupted text) |
| Document parsers (Unstructured, Docling, PyMuPDF) | Extract text, tables and layout from PDF, HTML, Office | Messy real-world files | Heavy dependencies; layout mistakes |
| Plain json / csv reading | No LangChain at all | Trivial data | You build the Documents yourself |

## This step's code

**When to use**

- Bringing any data into LangChain as Documents (text + metadata).
- No maintained loader exists for your source: write a BaseLoader with just lazy_load().

**Limitations**

- load() keeps every Document in memory; iterate lazy_load() for very large sources.
- Most classic loaders live in the sunset langchain-community package.
- The metadata you keep here decides what you can filter (step 18) and cite (step 13) later.
