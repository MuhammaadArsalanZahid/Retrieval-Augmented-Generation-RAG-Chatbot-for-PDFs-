"""test_retrieval.py - Step 7: test retrieval WITHOUT any interface.

Run:  python test_retrieval.py

For every question it checks whether a chunk from the correct page appears in
the top 3 results. At the end it prints:
  - the hit rate (top-3 accuracy)
  - a table you can paste into your README
  - a suggested relevance threshold for the app (using the 'not in PDF' questions)

Change CHUNK_SIZE / OVERLAP / TOP_K below and run again to see the effect.
"""
import os
from loader import load_pdf, chunk_pages
from vector_store import VectorStore, load_embedder

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
DOCS = "sample_docs"

LIB = "library_policy_sample.pdf"   # 4 pages, simple text
LONG = "long_report_sample.pdf"     # 55 pages
TAB = "tables_sample.pdf"           # 3 pages, answers are inside tables

# (pdf file, question, page(s) where the answer is)
TEST_CASES = [
    (LIB, "What are the library opening hours on weekdays?", [1]),
    (LIB, "How many books can an undergraduate student borrow?", [2]),
    (LIB, "What is the late return fine per day?", [3]),
    (LIB, "How many group study rooms are there?", [4]),
    (LONG, "How many million litres of water does the treatment plant process per day?", [7]),
    (LONG, "How long is the Eastern Bypass road?", [18]),
    (LONG, "How many public schools does the city operate?", [29]),
    (LONG, "How many tonnes of solid waste are collected daily?", [41]),
    (LONG, "What is the total approved budget for fiscal year 2026?", [52]),
    (TAB, "What is the price of the Mechanical Keyboard?", [1]),
    (TAB, "How many units of the USB-C Hub are in stock?", [1]),
    (TAB, "What grade did Usman Ali get?", [2]),
    (TAB, "Which region had the highest total sales?", [3]),
]

# Questions whose answer is NOT in the PDF (the app should say "not found")
NEGATIVE = [
    (LIB, "Who is the vice chancellor of the university?"),
    (LIB, "How much is the tuition fee per semester?"),
    (LIB, "Does the library have a cafeteria?"),
    (LONG, "Who won the cricket world cup?"),
    (TAB, "What is the phone number of the company?"),
]

_stores = {}


def get_store(embedder, pdf):
    """Build (once) the FAISS index for a PDF, same steps as the app."""
    if pdf not in _stores:
        with open(os.path.join(DOCS, pdf), "rb") as f:
            pages = load_pdf(f, pdf)
        chunks = chunk_pages(pages, CHUNK_SIZE, OVERLAP)
        store = VectorStore(embedder)
        store.build(chunks)
        _stores[pdf] = store
        print(f"[{pdf}] {len(pages)} pages -> {len(chunks)} chunks")
    return _stores[pdf]


def main():
    embedder = load_embedder()
    hits, rows, top_scores = 0, [], []

    print("\n=== QUESTIONS THAT ARE IN THE PDF ===")
    for i, (pdf, q, expected) in enumerate(TEST_CASES, start=1):
        results = get_store(embedder, pdf).search(q, k=TOP_K)
        pages = [c["page"] for c, _ in results]
        rank = next((r + 1 for r, p in enumerate(pages) if p in expected), None)
        hits += rank is not None
        top_scores.append(results[0][1])
        status = f"HIT (rank {rank})" if rank else "MISS"
        print(f"\n{i}. {q}\n   expected page {expected} -> {status}")
        for c, s in results:
            print(f"   [{s:.2f}] p.{c['page']}: {c['text'][:90]}...")
        rows.append((i, pdf, q, expected, status))

    print("\n=== QUESTIONS THAT ARE NOT IN THE PDF ===")
    neg_scores = []
    for q_pdf, q in NEGATIVE:
        c, s = get_store(embedder, q_pdf).search(q, k=1)[0]
        neg_scores.append(s)
        print(f"[{s:.2f}] {q}   ({q_pdf})")

    print(f"\n=== SUMMARY ===\nTop-{TOP_K} hit rate: {hits}/{len(TEST_CASES)} "
          f"({100 * hits / len(TEST_CASES):.0f}%)  [chunk={CHUNK_SIZE}, overlap={OVERLAP}]")
    lo, hi = max(neg_scores), min(top_scores)
    if lo < hi:
        print(f"Suggested threshold: about {(lo + hi) / 2:.2f} "
              f"(highest 'not in PDF' score {lo:.2f} < lowest real best score {hi:.2f})")
    else:
        print(f"Scores overlap (highest 'not in PDF' {lo:.2f} >= lowest real best {hi:.2f}). "
              "Pick a threshold carefully and write this limitation in your README.")

    print("\n=== README TABLE (copy) ===")
    print("| # | PDF | Question | Expected page | Result |\n|---|---|---|---|---|")
    for i, pdf, q, exp, st in rows:
        print(f"| {i} | {pdf} | {q} | {exp} | {st} |")


if __name__ == "__main__":
    main()
