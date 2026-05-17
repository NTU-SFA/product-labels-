import os
import re
import json
import time
import random
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"
TEMPERATURE = 0


# 例如 100 / 200
SAMPLE_PER_GROUP = 200
RANDOM_SEED = 42

HAZARD_DIR = "/Users/xueluangong/Desktop/GPT Source Codes/hazard"
CODE_DIR = "/Users/xueluangong/Desktop/GPT Source Codes/Assay_attr_Extraction_Codes"

HAS_HAZARDS_LABELLED_PATH = os.path.join(HAZARD_DIR, "rasff_data_2020_to_2026_has_hazards_with_labels.json")
NO_HAZARDS_LABELLED_PATH = os.path.join(HAZARD_DIR, "rasff_data_2020_to_2026_no_hazards_with_labels.json")

HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH = os.path.join(HAZARD_DIR, "has_hazards_mapping_hazard_label.json")
HAS_HAZARDS_MAPPING_HAZARD_CATEGORY_LABEL_PATH = os.path.join(HAZARD_DIR, "has_hazards_mapping_hazard_category_label.json")
NO_HAZARDS_MAPPING1_PATH = os.path.join(HAZARD_DIR, "no_hazards_mapping1.json")
NO_HAZARDS_MAPPING2_PATH = os.path.join(HAZARD_DIR, "no_hazards_mapping2.json")

PROMPT_PATH = os.path.join(CODE_DIR, "hazard_prompt_V2.txt")

OUTPUT_DIR = os.path.join(CODE_DIR, "Outputs_hazard_eval_mapping_first")
PREDICTION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hazard_eval_predictions_mapping_first.json")
ERROR_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hazard_eval_errors_mapping_first.json")
WRONG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hazard_eval_wrong_cases_mapping_first.json")
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hazard_eval_summary_mapping_first.json")


# =========================
# IO
# =========================
def load_json(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text(fp: str) -> str:
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


# =========================
# Utilities
# =========================
def normalize_labels(labels: List[str]) -> List[str]:
    if not isinstance(labels, list):
        return []
    out = []
    for x in labels:
        if isinstance(x, str):
            x = x.strip()
            if x:
                out.append(x)
    return sorted(list(set(out)))

def exact_match(pred: List[str], gold: List[str]) -> int:
    return int(normalize_labels(pred) == normalize_labels(gold))

def parse_json_output(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"Cannot parse JSON: {text}")

def sample_data(data: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n > len(data):
        raise ValueError(f"Requested sample size {n} > dataset size {len(data)}")
    rng = random.Random(seed)
    return rng.sample(data, n)

def build_chain(prompt_path: str):
    prompt_text = load_text(prompt_path)
    prompt = PromptTemplate(
        input_variables=[
            "allowed_hazard_labels",
            "allowed_hazard_category_labels",
            "subject",
            "unresolved_items_json"
        ],
        template=prompt_text
    )

    llm = ChatOpenAI(
        model_name=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE
    )
    return prompt | llm

def normalize_text_for_match(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s\(\)\+]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_for_ngrams(text: str) -> List[str]:
    text = normalize_text_for_match(text)
    return text.split() if text else []

def generate_ngrams(tokens: List[str], max_n: int = 5) -> List[str]:
    out = []
    seen = set()
    for n in range(max_n, 0, -1):
        for i in range(len(tokens) - n + 1):
            ng = " ".join(tokens[i:i+n])
            if ng not in seen:
                seen.add(ng)
                out.append(ng)
    return out

def extract_hazard_attribute(raw_hazard: str) -> str:
    if not isinstance(raw_hazard, str):
        return ""
    raw_hazard = raw_hazard.strip()
    if not raw_hazard:
        return ""
    return raw_hazard.split(" - ")[0].strip()


SPECIAL_CANONICAL_LABELS = {
    "salmonella spp": "Salmonella spp",
    "salmonella": "Salmonella spp",
    "polycyclic aromatic hydrocarbons sum of": "Polycyclic aromatic hydrocarbons",
    "polycyclic aromatic hydrocarbons": "Polycyclic aromatic hydrocarbons",
    "aflatoxins": "Aflatoxins",
    "aflatoxin b1": "Aflatoxins",
    "aflatoxin total": "Aflatoxins",
    "novel food ingredient": "Novel food ingredient",
    "suffocation risk consumption": "Suffocation risk",
    "organoleptic characteristics": "Organoleptic aspects",
}

def canonicalize_one_label(x: str, allowed_lookup: Dict[str, str], allowed_labels: List[str]) -> str:
    key = normalize_text_for_match(x)

    if key in SPECIAL_CANONICAL_LABELS:
        return SPECIAL_CANONICAL_LABELS[key]

    if key in allowed_lookup:
        return allowed_lookup[key]

    e_match = re.search(r"\be\s*([0-9]{3,4}[a-z]?)\b", key)
    if e_match:
        e_code = f"e{e_match.group(1)}"
        matches = [y for y in allowed_labels if normalize_text_for_match(y).startswith(e_code)]
        if len(matches) == 1:
            return matches[0]

    prefix_matches = [y for y in allowed_labels if normalize_text_for_match(y).startswith(key)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    contain_matches = [y for y in allowed_labels if key and key in normalize_text_for_match(y)]
    if len(contain_matches) == 1:
        return contain_matches[0]

    return x.strip()

def canonicalize_pred_labels(pred: List[str], allowed_lookup: Dict[str, str], allowed_labels: List[str]) -> List[str]:
    out = []
    for x in pred:
        if isinstance(x, str) and x.strip():
            out.append(canonicalize_one_label(x, allowed_lookup, allowed_labels))
    return normalize_labels(out)


# =========================
# Allowed label space
# =========================
def build_allowed_label_sets(
    has_labelled: List[Dict[str, Any]],
    no_labelled: List[Dict[str, Any]],
    has_hazard_label_inventory: Dict[str, Any],
    has_hazard_category_inventory: Dict[str, Any],
    no_mapping1: Dict[str, Any],
    no_mapping2: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    hazard_labels = set()
    hazard_category_labels = set()

    for row in has_labelled + no_labelled:
        for x in row.get("hazard_label", []):
            if isinstance(x, str) and x.strip():
                hazard_labels.add(x.strip())
        for x in row.get("hazard_category_label", []):
            if isinstance(x, str) and x.strip():
                hazard_category_labels.add(x.strip())

    for k, v in has_hazard_label_inventory.items():
        if isinstance(k, str) and k.strip():
            hazard_labels.add(k.strip())
        if isinstance(v, str) and v.strip():
            hazard_labels.add(v.strip())

    for k, v in has_hazard_category_inventory.items():
        if isinstance(k, str) and k.strip():
            hazard_category_labels.add(k.strip())
        if isinstance(v, str) and v.strip():
            hazard_category_labels.add(v.strip())

    for mp in [no_mapping1, no_mapping2]:
        for _, v in mp.items():
            if isinstance(v, dict):
                hl = v.get("hazard_label", "")
                hc = v.get("hazard_category_label", "")
                if isinstance(hl, str) and hl.strip():
                    hazard_labels.add(hl.strip())
                if isinstance(hc, str) and hc.strip():
                    hazard_category_labels.add(hc.strip())

    for v in SPECIAL_CANONICAL_LABELS.values():
        hazard_labels.add(v)
        hazard_category_labels.add(v)

    for x in ["Novel food", "Food additives", "Pesticide residues", "Veterinary drug residues"]:
        hazard_category_labels.add(x)

    return sorted(hazard_labels), sorted(hazard_category_labels)


# =========================
# no_hazards: ngram -> mapping1 -> mapping2
# =========================
def normalize_mapping_value(v: Dict[str, Any]) -> Dict[str, List[str]]:
    hazard_label = v.get("hazard_label", "")
    hazard_category_label = v.get("hazard_category_label", "")

    if isinstance(hazard_label, str):
        hazard_label = [hazard_label] if hazard_label.strip() else []
    elif not isinstance(hazard_label, list):
        hazard_label = []

    if isinstance(hazard_category_label, str):
        hazard_category_label = [hazard_category_label] if hazard_category_label.strip() else []
    elif not isinstance(hazard_category_label, list):
        hazard_category_label = []

    return {
        "hazard_label": normalize_labels(hazard_label),
        "hazard_category_label": normalize_labels(hazard_category_label),
    }

def compile_ngram_mapping(mapping_dict: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    compiled = {}
    for k, v in mapping_dict.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        nk = normalize_text_for_match(k)
        if nk:
            compiled[nk] = normalize_mapping_value(v)
    return compiled

def find_allowed_candidate(term_candidates: List[str], allowed_lookup: Dict[str, str], allowed_labels: List[str]) -> str:
    for t in term_candidates:
        nt = normalize_text_for_match(t)
        if nt in allowed_lookup:
            return allowed_lookup[nt]
    for t in term_candidates:
        nt = normalize_text_for_match(t)
        matches = [x for x in allowed_labels if nt and nt in normalize_text_for_match(x)]
        if len(matches) == 1:
            return matches[0]
    return ""

def postprocess_no_hazards(
    subject: str,
    hazard_labels: List[str],
    hazard_category_labels: List[str],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[List[str], List[str]]:
    text = normalize_text_for_match(subject)

    if "ethylene oxide" in text:
        eth = find_allowed_candidate(["Ethylene oxide"], allowed_hazard_lookup, allowed_hazard_labels)
        pr = find_allowed_candidate(["Pesticide residues"], allowed_hazard_category_lookup, allowed_hazard_category_labels)
        if eth:
            hazard_labels = sorted(set(hazard_labels + [eth]))
        if pr:
            hazard_category_labels = [pr]

    if any(x in text for x in ["foreign body", "piece of glass", "glass fragment", "plastic fragment", "plastic particles"]):
        fb = find_allowed_candidate(["Foreign bodies"], allowed_hazard_lookup, allowed_hazard_labels)
        fbc = find_allowed_candidate(["Foreign bodies"], allowed_hazard_category_lookup, allowed_hazard_category_labels)
        if fb:
            hazard_labels = [fb]
        if fbc:
            hazard_category_labels = [fbc]

    if "norovirus" in text:
        nv = find_allowed_candidate(["Norovirus"], allowed_hazard_lookup, allowed_hazard_labels)
        nvc = find_allowed_candidate(["Norovirus"], allowed_hazard_category_lookup, allowed_hazard_category_labels)
        if nv:
            hazard_labels = sorted(set(hazard_labels + [nv]))
        if nvc:
            hazard_category_labels = sorted(set(hazard_category_labels + [nvc]))

    if "plasticizer" in text:
        pl = find_allowed_candidate(["Plasticizer", "Plasticizers"], allowed_hazard_lookup, allowed_hazard_labels)
        plc = find_allowed_candidate(["Plasticizer", "Plasticizers"], allowed_hazard_category_lookup, allowed_hazard_category_labels)
        if pl:
            hazard_labels = sorted(set(hazard_labels + [pl]))
        if plc:
            hazard_category_labels = sorted(set(hazard_category_labels + [plc]))

    if any(x in text for x in ["cold chain", "breakage of the cold chain", "poor temperature control", "temperature failure"]):
        tc = find_allowed_candidate(["Temperature control"], allowed_hazard_lookup, allowed_hazard_labels)
        tcc = find_allowed_candidate(["Temperature control"], allowed_hazard_category_lookup, allowed_hazard_category_labels)
        if tc:
            hazard_labels = sorted(set(hazard_labels + [tc]))
        if tcc:
            hazard_category_labels = sorted(set(hazard_category_labels + [tcc]))

    return normalize_labels(hazard_labels), normalize_labels(hazard_category_labels)

def infer_no_hazards_from_subject(
    subject: str,
    mapping1: Dict[str, Dict[str, List[str]]],
    mapping2: Dict[str, Dict[str, List[str]]],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    tokens = tokenize_for_ngrams(subject)
    ngrams = generate_ngrams(tokens, max_n=5)

    hazard_labels = []
    hazard_category_labels = []
    matched_mapping1 = []
    matched_mapping2 = []

    for ng in ngrams:
        if ng in mapping1:
            matched_mapping1.append(ng)
            hazard_labels.extend(mapping1[ng]["hazard_label"])
            hazard_category_labels.extend(mapping1[ng]["hazard_category_label"])

    for ng in ngrams:
        if ng in mapping2:
            matched_mapping2.append(ng)
            hazard_labels.extend(mapping2[ng]["hazard_label"])
            hazard_category_labels.extend(mapping2[ng]["hazard_category_label"])

    hazard_labels = canonicalize_pred_labels(hazard_labels, allowed_hazard_lookup, allowed_hazard_labels)
    hazard_category_labels = canonicalize_pred_labels(
        hazard_category_labels, allowed_hazard_category_lookup, allowed_hazard_category_labels
    )

    hazard_labels, hazard_category_labels = postprocess_no_hazards(
        subject=subject,
        hazard_labels=hazard_labels,
        hazard_category_labels=hazard_category_labels,
        allowed_hazard_lookup=allowed_hazard_lookup,
        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
        allowed_hazard_labels=allowed_hazard_labels,
        allowed_hazard_category_labels=allowed_hazard_category_labels,
    )

    return hazard_labels, hazard_category_labels, matched_mapping1, matched_mapping2


# =========================
# has_hazards: attribute mapping first
# =========================
def infer_label_from_attr(attr: str, gold_labels: List[str], allowed_hazard_lookup: Dict[str, str], allowed_hazard_labels: List[str]) -> str:
    nattr = normalize_text_for_match(attr)

    if "aflatoxin" in nattr and "Aflatoxins" in gold_labels:
        return "Aflatoxins"
    if nattr in {"salmonella", "salmonella spp", "salmonella spp."} and "Salmonella spp" in gold_labels:
        return "Salmonella spp"

    can_attr = canonicalize_one_label(attr, allowed_hazard_lookup, allowed_hazard_labels)
    if can_attr in gold_labels:
        return can_attr
    if attr in gold_labels:
        return attr
    if len(gold_labels) == 1:
        return gold_labels[0]
    return ""

def infer_category_from_attr(
    attr: str,
    raw_category: str,
    gold_categories: List[str],
    chosen_label: str,
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_category_labels: List[str],
) -> str:
    nattr = normalize_text_for_match(attr)
    nrawcat = normalize_text_for_match(raw_category)

    if "aflatoxin" in nattr and "Aflatoxins" in gold_categories:
        return "Aflatoxins"
    if nattr in {"salmonella", "salmonella spp", "salmonella spp."} and "Salmonella spp" in gold_categories:
        return "Salmonella spp"
    if "salmonella " in nattr and "Salmonella spp" in gold_categories:
        return "Salmonella spp"
    if "pesticide residues" in nrawcat and "Pesticide residues" in gold_categories:
        return "Pesticide residues"
    if "residues of veterinary medicinal products" in nrawcat and "Veterinary drug residues" in gold_categories:
        return "Veterinary drug residues"
    if "novel food" in nrawcat and "Novel food" in gold_categories:
        return "Novel food"
    if "food additives" in nrawcat and "Food additives" in gold_categories:
        return "Food additives"

    if chosen_label and chosen_label in gold_categories:
        return chosen_label

    can_cat = canonicalize_one_label(raw_category, allowed_hazard_category_lookup, allowed_hazard_category_labels)
    if can_cat in gold_categories:
        return can_cat
    if len(gold_categories) == 1:
        return gold_categories[0]
    return ""

def build_has_hazards_attribute_maps(
    has_labelled: List[Dict[str, Any]],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    attr_to_label_counts = defaultdict(Counter)
    attr_to_cat_counts = defaultdict(Counter)

    for row in has_labelled:
        gold_labels = normalize_labels(row.get("hazard_label", []))
        gold_categories = normalize_labels(row.get("hazard_category_label", []))
        hazards = row.get("hazards", [])
        if not isinstance(hazards, list):
            continue

        for h in hazards:
            if not isinstance(h, dict):
                continue

            raw_hazard = h.get("hazard", "")
            raw_category = h.get("category", "")
            attr = extract_hazard_attribute(raw_hazard)
            nattr = normalize_text_for_match(attr)
            if not nattr:
                continue

            chosen_label = infer_label_from_attr(attr, gold_labels, allowed_hazard_lookup, allowed_hazard_labels)
            if chosen_label:
                attr_to_label_counts[nattr][chosen_label] += 1

            chosen_cat = infer_category_from_attr(
                attr, raw_category, gold_categories, chosen_label,
                allowed_hazard_category_lookup, allowed_hazard_category_labels
            )
            if chosen_cat:
                attr_to_cat_counts[nattr][chosen_cat] += 1

    attr_to_label = {k: cnt.most_common(1)[0][0] for k, cnt in attr_to_label_counts.items()}
    attr_to_cat = {k: cnt.most_common(1)[0][0] for k, cnt in attr_to_cat_counts.items()}
    return attr_to_label, attr_to_cat

def map_has_hazard_item(
    raw_hazard: str,
    raw_category: str,
    attr_to_label: Dict[str, str],
    attr_to_cat: Dict[str, str],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[List[str], List[str], bool]:
    attr = extract_hazard_attribute(raw_hazard)
    nattr = normalize_text_for_match(attr)

    labels = []
    categories = []

    if nattr in attr_to_label:
        labels = [attr_to_label[nattr]]
    if nattr in attr_to_cat:
        categories = [attr_to_cat[nattr]]

    if not labels:
        ncat = normalize_text_for_match(raw_category)

        if "aflatoxin" in nattr:
            labels = ["Aflatoxins"]
            categories = ["Aflatoxins"]
        elif nattr in {"salmonella", "salmonella spp", "salmonella spp."}:
            labels = ["Salmonella spp"]
            categories = ["Salmonella spp"]
        else:
            can_attr = canonicalize_one_label(attr, allowed_hazard_lookup, allowed_hazard_labels)
            if can_attr and can_attr != attr:
                labels = [can_attr]

            if not categories:
                if "pesticide residues" in ncat:
                    categories = ["Pesticide residues"]
                elif "residues of veterinary medicinal products" in ncat:
                    categories = ["Veterinary drug residues"]
                elif "novel food" in ncat:
                    categories = ["Novel food"]
                elif "food additives" in ncat:
                    categories = ["Food additives"]
                elif labels and labels[0] in allowed_hazard_category_labels:
                    categories = [labels[0]]

    labels = canonicalize_pred_labels(labels, allowed_hazard_lookup, allowed_hazard_labels)
    categories = canonicalize_pred_labels(categories, allowed_hazard_category_lookup, allowed_hazard_category_labels)

    resolved = bool(labels or categories)
    return labels, categories, resolved

def postprocess_has_hazards(
    subject: str,
    hazards: List[Dict[str, Any]],
    hazard_labels: List[str],
    hazard_category_labels: List[str],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[List[str], List[str]]:
    raw_hazards = [normalize_text_for_match(h.get("hazard", "")) for h in hazards if isinstance(h, dict)]
    raw_categories = [normalize_text_for_match(h.get("category", "")) for h in hazards if isinstance(h, dict)]

    hazard_labels = canonicalize_pred_labels(hazard_labels, allowed_hazard_lookup, allowed_hazard_labels)
    hazard_category_labels = canonicalize_pred_labels(
        hazard_category_labels, allowed_hazard_category_lookup, allowed_hazard_category_labels
    )

    if any("aflatoxin b1" in x for x in raw_hazards) or any("aflatoxin total" in x for x in raw_hazards):
        hazard_labels = ["Aflatoxins"]
        hazard_category_labels = ["Aflatoxins"]

    if any(x.startswith("salmonella") for x in raw_hazards):
        specific = [x for x in raw_hazards if "salmonella " in x and "salmonella spp" not in x and "salmonella -" not in x]
        if not specific:
            hazard_labels = ["Salmonella spp"]
            hazard_category_labels = ["Salmonella spp"]

    if any("polycyclic aromatic hydrocarbons sum of" in x for x in raw_hazards):
        hazard_labels = [x for x in hazard_labels if normalize_text_for_match(x) != "polycyclic aromatic hydrocarbons sum of"]
        if "Polycyclic aromatic hydrocarbons" not in hazard_labels:
            hazard_labels.append("Polycyclic aromatic hydrocarbons")

    if any("novel food ingredient" in x for x in raw_hazards):
        hazard_labels = ["Novel food ingredient"]
        hazard_category_labels = ["Novel food"]

    if any("residues of veterinary medicinal products" in x for x in raw_categories):
        hazard_category_labels = ["Veterinary drug residues"]

    if "Sulphite" in hazard_labels:
        hazard_category_labels = ["Food additives"]

    if not hazard_category_labels:
        for x in hazard_labels:
            if x in allowed_hazard_category_labels:
                hazard_category_labels.append(x)

    return normalize_labels(hazard_labels), normalize_labels(hazard_category_labels)


# =========================
# Fallback prompt for unresolved has_hazards items
# =========================
def resolve_unmapped_has_hazards_with_prompt(
    subject: str,
    unresolved_items: List[Dict[str, str]],
    chain,
    allowed_hazard_labels_str: str,
    allowed_hazard_category_labels_str: str,
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_labels: List[str],
) -> Dict[str, Dict[str, List[str]]]:
    if not unresolved_items:
        return {}

    response = chain.invoke({
        "allowed_hazard_labels": allowed_hazard_labels_str,
        "allowed_hazard_category_labels": allowed_hazard_category_labels_str,
        "subject": subject,
        "unresolved_items_json": json.dumps(unresolved_items, ensure_ascii=False)
    })

    raw_output = str(response.content)
    parsed = parse_json_output(raw_output)

    out = {}
    for item in parsed.get("items", []):
        if not isinstance(item, dict):
            continue
        raw_attr = item.get("raw_hazard_attribute", "")
        if not isinstance(raw_attr, str) or not raw_attr.strip():
            continue

        key = normalize_text_for_match(raw_attr)
        labels = canonicalize_pred_labels(item.get("hazard_label", []), allowed_hazard_lookup, allowed_hazard_labels)
        cats = canonicalize_pred_labels(item.get("hazard_category_label", []), allowed_hazard_category_lookup, allowed_hazard_category_labels)

        out[key] = {
            "hazard_label": labels,
            "hazard_category_label": cats
        }

    return out


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")

    has_data = load_json(HAS_HAZARDS_LABELLED_PATH)
    no_data = load_json(NO_HAZARDS_LABELLED_PATH)

    has_hazard_label_inventory = load_json(HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH)
    has_hazard_category_inventory = load_json(HAS_HAZARDS_MAPPING_HAZARD_CATEGORY_LABEL_PATH)
    no_mapping1 = load_json(NO_HAZARDS_MAPPING1_PATH)
    no_mapping2 = load_json(NO_HAZARDS_MAPPING2_PATH)

    allowed_hazard_labels, allowed_hazard_category_labels = build_allowed_label_sets(
        has_labelled=has_data,
        no_labelled=no_data,
        has_hazard_label_inventory=has_hazard_label_inventory,
        has_hazard_category_inventory=has_hazard_category_inventory,
        no_mapping1=no_mapping1,
        no_mapping2=no_mapping2,
    )

    allowed_hazard_set = set(allowed_hazard_labels)
    allowed_hazard_category_set = set(allowed_hazard_category_labels)

    allowed_hazard_lookup = {normalize_text_for_match(x): x for x in allowed_hazard_labels}
    allowed_hazard_category_lookup = {normalize_text_for_match(x): x for x in allowed_hazard_category_labels}

    mapping1 = compile_ngram_mapping(no_mapping1)
    mapping2 = compile_ngram_mapping(no_mapping2)

    attr_to_label, attr_to_cat = build_has_hazards_attribute_maps(
        has_labelled=has_data,
        allowed_hazard_lookup=allowed_hazard_lookup,
        allowed_hazard_labels=allowed_hazard_labels,
        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
        allowed_hazard_category_labels=allowed_hazard_category_labels,
    )

    sampled_has = sample_data(has_data, SAMPLE_PER_GROUP, RANDOM_SEED)
    sampled_no = sample_data(no_data, SAMPLE_PER_GROUP, RANDOM_SEED + 1)

    eval_pool = []
    for row in sampled_has:
        x = dict(row)
        x["_group"] = "has_hazards"
        eval_pool.append(x)
    for row in sampled_no:
        x = dict(row)
        x["_group"] = "no_hazards"
        eval_pool.append(x)

    chain = build_chain(PROMPT_PATH)
    allowed_hazard_labels_str = json.dumps(allowed_hazard_labels, ensure_ascii=False)
    allowed_hazard_category_labels_str = json.dumps(allowed_hazard_category_labels, ensure_ascii=False)

    results = []
    errors = []
    wrong_cases = []

    overall = {
        "total": 0,
        "hazard_label_correct": 0,
        "hazard_category_label_correct": 0,
        "joint_correct": 0
    }

    per_group = {
        "has_hazards": {"total": 0, "hazard_label_correct": 0, "hazard_category_label_correct": 0, "joint_correct": 0},
        "no_hazards": {"total": 0, "hazard_label_correct": 0, "hazard_category_label_correct": 0, "joint_correct": 0},
    }

    source_stats = {
        "has_hazards_mapping": 0,
        "has_hazards_prompt_fallback": 0,
        "no_hazards_mapping1_hits": 0,
        "no_hazards_mapping2_hits": 0,
        "no_hazards_no_match": 0,
    }

    for idx, item in enumerate(tqdm(eval_pool, desc="Evaluating hazard mapping-first")):
        time.sleep(0.01)
        group = item["_group"]

        try:
            subject = item.get("subject", "")
            hazards = item.get("hazards", [])

            if group == "has_hazards":
                pred_hazard_labels = []
                pred_hazard_categories = []
                unresolved_items = []

                for h in hazards:
                    if not isinstance(h, dict):
                        continue
                    raw_hazard = h.get("hazard", "")
                    raw_category = h.get("category", "")

                    labels, cats, resolved = map_has_hazard_item(
                        raw_hazard=raw_hazard,
                        raw_category=raw_category,
                        attr_to_label=attr_to_label,
                        attr_to_cat=attr_to_cat,
                        allowed_hazard_lookup=allowed_hazard_lookup,
                        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
                        allowed_hazard_labels=allowed_hazard_labels,
                        allowed_hazard_category_labels=allowed_hazard_category_labels,
                    )

                    if resolved:
                        source_stats["has_hazards_mapping"] += 1
                        pred_hazard_labels.extend(labels)
                        pred_hazard_categories.extend(cats)
                    else:
                        unresolved_items.append({
                            "raw_hazard_attribute": extract_hazard_attribute(raw_hazard),
                            "raw_category": raw_category
                        })

                prediction_source = "has_hazards_mapping_first"

                if unresolved_items:
                    prediction_source = "has_hazards_mapping_plus_prompt_fallback"
                    source_stats["has_hazards_prompt_fallback"] += len(unresolved_items)

                    fallback_res = resolve_unmapped_has_hazards_with_prompt(
                        subject=subject,
                        unresolved_items=unresolved_items,
                        chain=chain,
                        allowed_hazard_labels_str=allowed_hazard_labels_str,
                        allowed_hazard_category_labels_str=allowed_hazard_category_labels_str,
                        allowed_hazard_lookup=allowed_hazard_lookup,
                        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
                        allowed_hazard_labels=allowed_hazard_labels,
                        allowed_hazard_category_labels=allowed_hazard_category_labels,
                    )

                    for u in unresolved_items:
                        key = normalize_text_for_match(u["raw_hazard_attribute"])
                        if key in fallback_res:
                            pred_hazard_labels.extend(fallback_res[key]["hazard_label"])
                            pred_hazard_categories.extend(fallback_res[key]["hazard_category_label"])

                pred_hazard_labels, pred_hazard_categories = postprocess_has_hazards(
                    subject=subject,
                    hazards=hazards,
                    hazard_labels=pred_hazard_labels,
                    hazard_category_labels=pred_hazard_categories,
                    allowed_hazard_lookup=allowed_hazard_lookup,
                    allowed_hazard_category_lookup=allowed_hazard_category_lookup,
                    allowed_hazard_labels=allowed_hazard_labels,
                    allowed_hazard_category_labels=allowed_hazard_category_labels,
                )

                pred_hazard_labels = [x for x in pred_hazard_labels if x in allowed_hazard_set]
                pred_hazard_categories = [x for x in pred_hazard_categories if x in allowed_hazard_category_set]

                matched_keys = []
                raw_output = json.dumps({
                    "hazard_label": pred_hazard_labels,
                    "hazard_category_label": pred_hazard_categories
                }, ensure_ascii=False)

            else:
                pred_hazard_labels, pred_hazard_categories, matched_mapping1, matched_mapping2 = infer_no_hazards_from_subject(
                    subject=subject,
                    mapping1=mapping1,
                    mapping2=mapping2,
                    allowed_hazard_lookup=allowed_hazard_lookup,
                    allowed_hazard_category_lookup=allowed_hazard_category_lookup,
                    allowed_hazard_labels=allowed_hazard_labels,
                    allowed_hazard_category_labels=allowed_hazard_category_labels,
                )

                pred_hazard_labels = [x for x in pred_hazard_labels if x in allowed_hazard_set]
                pred_hazard_categories = [x for x in pred_hazard_categories if x in allowed_hazard_category_set]

                source_stats["no_hazards_mapping1_hits"] += len(set(matched_mapping1))
                source_stats["no_hazards_mapping2_hits"] += len(set(matched_mapping2))
                if not pred_hazard_labels and not pred_hazard_categories:
                    source_stats["no_hazards_no_match"] += 1

                prediction_source = "no_hazards_mapping_ngram"
                matched_keys = sorted(list(set(matched_mapping1 + matched_mapping2)))
                raw_output = json.dumps({
                    "hazard_label": pred_hazard_labels,
                    "hazard_category_label": pred_hazard_categories
                }, ensure_ascii=False)

            gold_hazard_label = normalize_labels(item.get("hazard_label", []))
            gold_hazard_category_label = normalize_labels(item.get("hazard_category_label", []))

            match_hazard = exact_match(pred_hazard_labels, gold_hazard_label)
            match_category = exact_match(pred_hazard_categories, gold_hazard_category_label)
            joint_match = int(match_hazard == 1 and match_category == 1)

            overall["total"] += 1
            overall["hazard_label_correct"] += match_hazard
            overall["hazard_category_label_correct"] += match_category
            overall["joint_correct"] += joint_match

            per_group[group]["total"] += 1
            per_group[group]["hazard_label_correct"] += match_hazard
            per_group[group]["hazard_category_label_correct"] += match_category
            per_group[group]["joint_correct"] += joint_match

            out_item = dict(item)
            out_item["predicted_hazard_label"] = pred_hazard_labels
            out_item["predicted_hazard_category_label"] = pred_hazard_categories
            out_item["match_hazard_label"] = match_hazard
            out_item["match_hazard_category_label"] = match_category
            out_item["joint_match"] = joint_match
            out_item["prediction_source"] = prediction_source
            out_item["matched_mapping_keys"] = matched_keys
            out_item["raw_output"] = raw_output
            results.append(out_item)

            if joint_match == 0:
                wrong_cases.append(out_item)

            if idx < 5:
                print("\n[Sample Prediction]")
                print(json.dumps({
                    "group": group,
                    "source": prediction_source,
                    "reference": item.get("reference", ""),
                    "subject": subject,
                    "gold_hazard_label": gold_hazard_label,
                    "pred_hazard_label": pred_hazard_labels,
                    "gold_hazard_category_label": gold_hazard_category_label,
                    "pred_hazard_category_label": pred_hazard_categories,
                    "joint_match": joint_match,
                    "matched_mapping_keys": matched_keys
                }, ensure_ascii=False, indent=2))

        except Exception as e:
            err = {
                "group": group,
                "reference": item.get("reference", ""),
                "subject": item.get("subject", ""),
                "error": str(e)
            }
            errors.append(err)

            if len(errors) <= 5:
                print("\n[Sample Error]")
                print(json.dumps(err, ensure_ascii=False, indent=2))

    summary = {
        "sample_per_group": SAMPLE_PER_GROUP,
        "random_seed": RANDOM_SEED,
        "source_stats": source_stats,
        "overall": {
            "total": overall["total"],
            "hazard_label_accuracy": overall["hazard_label_correct"] / overall["total"] if overall["total"] else 0.0,
            "hazard_category_label_accuracy": overall["hazard_category_label_correct"] / overall["total"] if overall["total"] else 0.0,
            "joint_exact_match_accuracy": overall["joint_correct"] / overall["total"] if overall["total"] else 0.0,
            "errors": len(errors),
        },
        "per_group": {}
    }

    for group, stats in per_group.items():
        total = stats["total"]
        summary["per_group"][group] = {
            "total": total,
            "hazard_label_accuracy": stats["hazard_label_correct"] / total if total else 0.0,
            "hazard_category_label_accuracy": stats["hazard_category_label_correct"] / total if total else 0.0,
            "joint_exact_match_accuracy": stats["joint_correct"] / total if total else 0.0,
        }

    with open(PREDICTION_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(ERROR_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    with open(WRONG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(wrong_cases, f, indent=2, ensure_ascii=False)

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Overall Evaluation =====")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))

    print("\n===== Per-Group Evaluation =====")
    for group, stats in summary["per_group"].items():
        print(f"\n[{group}]")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n===== Source Stats =====")
    print(json.dumps(source_stats, ensure_ascii=False, indent=2))

    print(f"\nPredictions saved to: {PREDICTION_OUTPUT_PATH}")
    print(f"Wrong cases saved to: {WRONG_OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()