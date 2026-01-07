import os
import re
import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    pipeline
)
import torch


# ================== PATH CONFIG ==================
INPUT_DIR = "input"
OUTPUT_DIR = "claims_nlp_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================================================


# ================== MODELS ======================
NER_MODEL_NAME = "dslim/bert-base-NER"
LLM_MODEL_NAME = "microsoft/phi-2"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- NER ----
ner_pipe = pipeline(
    "ner",
    model=AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME),
    tokenizer=AutoTokenizer.from_pretrained(NER_MODEL_NAME),
    aggregation_strategy="simple"
)

# ---- LLM ----
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
llm_model.eval()
# =================================================


# -------- JSON SCHEMA --------
SCHEMA_JSON = """
{
  "patient": { "name": "", "age": "", "gender": "", "mrn": "" },
  "visit": { "admission_date": "", "discharge_date": "", "hospital_name": "", "doctor_name": "" },
  "clinical": {
    "diagnosis": [],
    "symptoms": [],
    "tests": [],
    "procedures": [],
    "medications": [],
    "treatment_summary": ""
  },
  "billing": { "bill_amount": "", "insurance_policy_number": "" }
}
"""
# =====================================================================


# -------- Chunk text --------
def chunk_text(text, size=1800, overlap=120):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


# -------- Batch NER --------
def run_ner(text):
    entities = []
    for out in ner_pipe(chunk_text(text), batch_size=4):
        entities.extend(out)
    return entities


# -------- Summarize NER for LLM --------
def summarize_entities(entities):
    return "\n".join(
        f"{e['entity_group']} -> {e['word']}"
        for e in entities[:120]
    )


# -------- LLM CALL --------
def call_llm(full_text, ner_entities):
    entity_summary = summarize_entities(ner_entities)

    prompt = f"""
You are a medical claims auditor.

RULES:
- Output ONE JSON object ONLY
- Follow schema exactly
- Leave blank if unsure
- Do NOT invent values
- No explanations, no extra text

DOCUMENT:
\"\"\"{full_text[:4500]}\"\"\"

NER HINTS:
{entity_summary}

Return JSON ONLY:

{SCHEMA_JSON}
"""

    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=450,
        do_sample=False,                 # deterministic
        use_cache=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )

    reply = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ---- extract ONLY valid JSON objects ----
    candidates = re.findall(r"\{[\s\S]*?\}", reply)

    for js in reversed(candidates):
        try:
            return json.loads(js)
        except:
            continue

    return {"error": "no valid JSON", "raw": reply}


# -------- PROCESS FILE --------
def process_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    ents = run_ner(text)
    result = call_llm(text, ents)

    out = Path(OUTPUT_DIR, path.stem + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✔ NLP extracted → {out}")


# -------- MAIN --------
def main():
    txt_files = list(Path(INPUT_DIR).rglob("*.txt"))

    if not txt_files:
        print("No text files found in input")
        return

    for f in txt_files:
        try:
            process_file(f)
        except Exception as e:
            print(f"❌ Failed {f}: {e}")


if __name__ == "__main__":
    main()
