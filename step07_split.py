"""Step 7: split documents into chunks.

An article (~1,750 words) is too long to embed as one vector and too long to put several into a
prompt. A text splitter cuts each Document into smaller chunks; each chunk is itself a Document
that keeps the article's metadata.

RecursiveCharacterTextSplitter tries to cut at paragraph breaks first, then lines, then spaces,
so chunks end at natural boundaries. `chunk_overlap` repeats the end of one chunk at the start of
the next, but only in whole pieces: if the cut falls at a paragraph longer than the overlap,
nothing is repeated. Here only about half of neighbouring chunks actually overlap.

Run:  uv run step07_split.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from step06_load import REVISION, MultiHopRAGLoader

articles = MultiHopRAGLoader(revision=REVISION).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # max characters per chunk (~150–200 words)
    chunk_overlap=200,    # characters shared between neighbouring chunks
    add_start_index=True, # records where each chunk starts in its article
)
chunks = splitter.split_documents(articles)

sizes = [len(chunk.page_content) for chunk in chunks]
print(f"{len(articles)} articles → {len(chunks)} chunks")
print(f"Chunk size in characters: min {min(sizes)} · avg {sum(sizes) // len(sizes)} · max {max(sizes)}")

print("\nMetadata is copied to every chunk, plus start_index:")
print(" ", chunks[1].metadata)

neighbours = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.metadata["url"] == b.metadata["url"]]
overlapping = [(a, b) for a, b in neighbours
               if b.metadata["start_index"] < a.metadata["start_index"] + len(a.page_content)]
print(f"\nNeighbouring chunks that really overlap: {len(overlapping)} of {len(neighbours)}")

first, second = overlapping[0]
print("Example: the end of one chunk …")
print("  …", first.page_content[-120:].replace("\n", " "))
print("… reappears at the start of the next:")
print(" ", second.page_content[:120].replace("\n", " "), "…")

tiny = [chunk for chunk in chunks if len(chunk.page_content) < 100]
print(f"\nTiny chunks (< 100 characters): {len(tiny)}, e.g. {tiny[0].page_content!r}")
