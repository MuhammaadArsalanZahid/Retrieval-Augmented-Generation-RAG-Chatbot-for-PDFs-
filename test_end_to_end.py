"""test_end_to_end.py - runs the FULL pipeline (retrieval + threshold + flan-t5 answer)
on 10 questions and prints a README-ready table with the app's real answers.

Run:  python test_end_to_end.py
Then copy the printed table into the README (section "End-to-end test").
The 'Correct?' column is only an automatic guess (keyword check): read every answer yourself and fix it if needed.
"""
import os
import re
from loader import load_pdf, chunk_pages
from vector_store import VectorStore, load_embedder
from generator import load_generator, generate_answer, NOT_FOUND

PDF = os.path.join("sample_docs", "library_policy_sample.pdf")
CHUNK_SIZE, OVERLAP, TOP_K = 500, 100, 3
THRESHOLD = 0.55   # use the same value as the app's threshold slider

# (question, in_pdf, regex that a correct answer should contain)
CASES = [
    ("What are the library opening hours on weekdays?", True, r"8:00"),
    ("How many books can a postgraduate student borrow?", True, r"\b8\b"),
    ("What is the maximum fine for one book?", True, r"500"),
    ("How many online research databases are available?", True, r"\b12\b"),
    ("Where is the library located?", True, r"science block|second floor"),
    ("What must a student leave at the desk when using a library laptop?", True, r"\bid\b|card"),
    ("Who decides the fine for a damaged book?", True, r"librarian"),
    ("Who is the vice chancellor of the university?", False, None),
    ("How much is the tuition fee per semester?", False, None),
    ("Does the library have a cafeteria?", False, None),
]


def clean(text):
    return text.replace("|", "/").replace("\n", " ").strip()


def main():
    embedder = load_embedder()
    tok, model = load_generator()
    with open(PDF, "rb") as f:
        pages = load_pdf(f, os.path.basename(PDF))
    store = VectorStore(embedder)
    store.build(chunk_pages(pages, CHUNK_SIZE, OVERLAP))

    rows, correct = [], 0
    for i, (q, in_pdf, pattern) in enumerate(CASES, start=1):
        results = store.search(q, k=TOP_K)
        if not results or results[0][1] < THRESHOLD:
            answer, pages_used = NOT_FOUND, []
        else:
            answer = generate_answer(q, [c for c, _ in results], tok, model)
            pages_used = sorted({c["page"] for c, _ in results}) if answer != NOT_FOUND else []
        if in_pdf:
            ok = answer != NOT_FOUND and re.search(pattern, answer.lower()) is not None
        else:
            ok = answer == NOT_FOUND
        correct += ok
        print(f"{i}. {q}\n   -> {answer}  (pages {pages_used})  [{'OK' if ok else 'CHECK'}]")
        rows.append((i, q, "Yes" if in_pdf else "No", clean(answer), "Yes" if ok else "No"))

    print("\n=== README TABLE (copy) ===")
    print("| # | Question | In PDF? | App answer | Correct? |\n|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(map(str, r)) + " |")
    print(f"\n**Result: {correct}/{len(CASES)} correct.** (automatic check, please verify manually)")


if __name__ == "__main__":
    main()
