"""generator.py - Prompt building and answer generation (Step 8)."""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

GEN_MODEL = "google/flan-t5-base"
MAX_INPUT_TOKENS = 480  # flan-t5-base reads ~512 tokens, keep a safety margin
NOT_FOUND = "Sorry, I could not find the answer to this question in the document."


def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    model.eval()
    return tok, model


def build_prompt(question, chunks, tokenizer):
    """Put as many retrieved chunks as fit into the token limit."""
    head = "Answer the question using only the context below. If the answer is not in the context, say 'not found'.\n\nContext:\n"
    tail = f"\n\nQuestion: {question}\nAnswer:"
    budget = MAX_INPUT_TOKENS - len(tokenizer.encode(head + tail))
    context = ""
    for c in chunks:
        piece = c["text"] + "\n"
        if len(tokenizer.encode(context + piece)) > budget:
            break
        context += piece
    return head + context + tail


def generate_answer(question, chunks, tokenizer, model):
    prompt = build_prompt(question, chunks, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
    answer = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if not answer or "not found" in answer.lower():
        return NOT_FOUND
    return answer
