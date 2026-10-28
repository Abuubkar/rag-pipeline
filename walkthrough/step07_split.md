# Step 7: split documents into chunks

Notes for [`step07_split.py`](step07_split.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ RecursiveCharacterTextSplitter | Cuts at paragraphs, then lines, then spaces | General prose; the usual default | Counts characters, not meaning; overlap is best-effort |
| Token-based (TokenTextSplitter, from_tiktoken_encoder) | Measures size in model tokens | Chunks must fit a model's token limit exactly | Needs a tokenizer; can cut mid-sentence |
| Sentence-based (NLTK / spaCy splitters) | Cuts only between sentences | Facts must stay whole | Extra dependency; uneven chunk sizes |
| Structure-aware (MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, code splitters) | Cuts at headings or code blocks; headings go into metadata | Docs, wikis, source code | Needs structured input; our news articles are plain text |
| Semantic chunking (SemanticChunker, langchain-experimental) | Embeds sentences and cuts where the topic changes | Long texts that drift between topics | Embeds every sentence (slow); experimental package |
| No splitting | One vector per whole document | Short documents: FAQs, tweets, product cards | A long article blurs into one vague vector |

## This step's code

**When to use**

- Documents longer than one embedding or one prompt should hold.
- Prose with paragraphs, where cutting at paragraph breaks keeps ideas together.

**Limitations**

- It cuts by characters, not meaning: a fact can be split across two chunks.
- chunk_overlap is best-effort: only 48% of neighbouring chunks actually overlapped.
- It leaves tiny fragments (53 chunks under 100 characters).
- chunk_size is found by trial and error; there's no right value for every corpus.
