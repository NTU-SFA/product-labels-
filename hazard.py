import os
import re
import json
import time
import random
import unicodedata
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

from tqdm import tqdm


# =========================
# Config
# =========================
SAMPLE_PER_GROUP = 1000
RANDOM_SEED = 42

HAZARD_DIR = "/Users/xueluangong/Desktop/GPT Source Codes/hazard"
CODE_DIR = "/Users/xueluangong/Desktop/GPT Source Codes/Assay_attr_Extraction_Codes"

HAS_HAZARDS_LABELLED_PATH = os.path.join(HAZARD_DIR, "rasff_data_2020_to_2026_has_hazards_with_labels.json")
NO_HAZARDS_LABELLED_PATH = os.path.join(HAZARD_DIR, "rasff_data_2020_to_2026_no_hazards_with_labels.json")

HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH = os.path.join(HAZARD_DIR, "has_hazards_mapping_hazard_label.json")
HAS_HAZARDS_MAPPING_HAZARD_CATEGORY_LABEL_PATH = os.path.join(HAZARD_DIR, "has_hazards_mapping_hazard_category_label.json")
NO_HAZARDS_MAPPING1_PATH = os.path.join(HAZARD_DIR, "no_hazards_mapping1.json")
NO_HAZARDS_MAPPING2_PATH = os.path.join(HAZARD_DIR, "no_hazards_mapping2.json")
MAPPING_LABEL_TO_CATEGORY_PATH = os.path.join(HAZARD_DIR, "mapping_hazard_label_to_hazard_category_label.json")

OUTPUT_DIR = os.path.join(CODE_DIR, "Outputs_hazard_eval_rule_only_v6")
PREDICTION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hazard_eval_predictions_rule_only_v6.json")
ERROR_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hazard_eval_errors_rule_only_v6.json")
WRONG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hazard_eval_wrong_cases_rule_only_v6.json")
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hazard_eval_summary_rule_only_v6.json")


# =========================
# IO
# =========================
def load_json(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Basic utils
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


def sample_data(data: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n > len(data):
        raise ValueError(f"Requested sample size {n} > dataset size {len(data)}")
    rng = random.Random(seed)
    return rng.sample(data, n)


def normalize_text_for_match(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()

    text = text.replace("sulfur", "sulphur")
    text = text.replace("flavourings", "flavorings")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")

    text = re.sub(r"[^a-z0-9\s\(\)\+\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def norm_key(text: str) -> str:
    return normalize_text_for_match(text).replace(".", "")


def tokenize(text: str) -> List[str]:
    t = normalize_text_for_match(text)
    return t.split() if t else []


def generate_ngrams(tokens: List[str], max_n: int = 5) -> List[str]:
    out = []
    seen = set()
    for n in range(max_n, 0, -1):
        for i in range(len(tokens) - n + 1):
            ng = " ".join(tokens[i:i + n])
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


def extract_ecodes(text: str) -> List[str]:
    """
    Capture:
    E102
    E 102
    E150d
    E 150d
    E450ii
    """
    if not text:
        return []
    t = normalize_text_for_match(text)
    matches = re.findall(r"\be\s*([0-9]{3,4}[a-z]{0,3})\b", t)
    out = [f"e{m}" for m in matches]
    return sorted(list(set(out)))


def regex_any(patterns: List[str], text: str) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False


# =========================
# Preferred canonical forms from manually labelled data
# =========================
def build_preferred_lookup_from_gold(rows: List[Dict[str, Any]], field: str) -> Dict[str, str]:
    counts = defaultdict(Counter)
    first_seen = {}

    idx = 0
    for row in rows:
        for x in row.get(field, []):
            if isinstance(x, str) and x.strip():
                k = norm_key(x)
                counts[k][x.strip()] += 1
                if k not in first_seen:
                    first_seen[k] = idx
                    idx += 1

    preferred = {}
    for k, c in counts.items():
        # highest frequency first; then shorter string; then lexicographic
        best = sorted(c.items(), key=lambda z: (-z[1], len(z[0]), z[0]))[0][0]
        preferred[k] = best
    return preferred


def merge_preferred_and_allowed(preferred_lookup: Dict[str, str], allowed_labels: List[str]) -> Dict[str, str]:
    merged = dict(preferred_lookup)
    for x in allowed_labels:
        k = norm_key(x)
        if k not in merged:
            merged[k] = x
    return merged


# =========================
# Canonicalization
# =========================
def canonicalize_one_label(
    x: str,
    allowed_lookup: Dict[str, str],
    allowed_labels: List[str]
) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""

    raw = x.strip()
    key = norm_key(raw)

    manual_aliases = {
        # Salmonella family
        "salmonella spp": "Salmonella spp",
        "salmonella spp.": "Salmonella spp",
        "salmonella": "Salmonella spp",
        "salmonella enteritidis": "Salmonella Enteritidis",
        "salmonella infantis": "Salmonella infantis",
        "salmonella derby": "Salmonella Derby",
        "salmonella agona": "Salmonella Agona",
        "salmonella mbandaka": "Salmonella Mbandaka",
        "salmonella newport": "Salmonella Newport",
        "salmonella anatum": "Salmonella anatum",
        "salmonella minnesota": "Salmonella Minnesota",
        "salmonella senftenberg": "Salmonella Senftenberg",
        "salmonella saintpaul": "Salmonella Saintpaul",
        "salmonella morehead": "Salmonella Morehead",
        "salmonella livingstone": "Salmonella Livingstone",
        "salmonella amsterdam": "Salmonella Amsterdam",
        "salmonella oranienburg": "Salmonella Oranienburg",
        "salmonella braenderup": "Salmonella Braenderup",
        "salmonella javiana": "Salmonella Javiana",
        "salmonella kiambu": "Salmonella Kiambu",
        "salmonella rubislaw": "Salmonella Rubislaw",
        "salmonella gaminara": "Salmonella gaminara",
        "salmonella glostrup": "Salmonella Glostrup",

        # Other pathogens
        "listeria monocytogenes": "Listeria monocytogenes",
        "listeria spp": "Listeria spp",
        "vibrio vulnificus": "Vibrio vulnificus",
        "vibrio spp": "Vibrio spp",
        "cronobacter sakazakii": "Cronobacter sakazakii",
        "cronobacter spp": "Cronobacter spp",
        "escherichia coli": "Escherichia coli",

        # Special strings
        "novel food ingredient": "Novel food ingredient",
        "novel food": "Novel food",
        "mould": "Mould",
        "moulds": "Moulds",
        "mold": "Mould",
        "molds": "Moulds",
        "nitrofuran metabolite furazolidone (aoz)": "Nitrofuran (metabolite) furazolidone (AOZ)",
        "nitrofuran metabolite furazolidone aoz": "Nitrofuran (metabolite) furazolidone (AOZ)",
        "chlorpyriphos ethyl": "Chlorpyriphos-ethyl",
        "chlorpyrifos ethyl": "Chlorpyriphos-ethyl",
        "fosthiazale": "Fosthiazale",
        "colour e 110": "Colour E 110",
        "e110 sunset yellow fcf": "E110 Sunset Yellow FCF",
        "e220 sulphur dioxide": "E220- sulfur dioxide",
        "e220 sulfur dioxide": "E220- sulfur dioxide",
        "organoleptic characteristics unsuitable": "Organoleptic aspects",
        "organoleptic aspects": "Organoleptic aspects",
        "ragweed (ambrosia spp.)": "Ragweed (Ambrosia spp)",
        "ragweed (ambrosia spp)": "Ragweed (Ambrosia spp)",
        "anisakis": "Anisakis",
        "insects larvae": "Insects",
        "aflatoxin total": "Aflatoxin total",
        "aflatoxin b1": "Aflatoxin B1",
        "aflatoxins": "Aflatoxins",
        "ergot alkaloids": "Ergot",
        "rye ergot": "Ergot",
        "polycyclic aromatic hydrocarbons": "Polycyclic aromatic hydrocarbons",
        "polycyclic aromatic hydrocarbons sum of": "Polycyclic aromatic hydrocarbons",
        "sulphite": "Sulphite",
        "food additives": "Food additives",
        "allergens": "Allergens",
        "documentation": "Documentation",
        "labelling": "Labelling",
        "foreign bodies": "Foreign bodies",
        "cannabinoid": "Cannabinoid",
        "vitamin b6": "Vitamin B6",
        "vitamin b3": "Vitamin B3",
        "vitamins": "Vitamins",
        "sibutramine": "Sibutramine",
        "antibiotics": "Antibiotics",
        "monacolin k": "Monacolin K",
        "parasitic infestation": "Parasitic infestation",
        "porcine dna": "Porcine DNA",
        "poultry dna": "Poultry DNA",
        "lead": "Lead",
        "ethylene oxide": "Ethylene oxide",
        "yohimbine": "Yohimbine",
        "veterinary control": "Veterinary control",
        "feed additives": "Feed additives",
        "temperature control": "Temperature control",
        "formaldehyde": "Formaldehyde",
        "bamboo": "Bamboo",
        "melamine": "Melamine",
    }

    if key in manual_aliases:
        alias = manual_aliases[key]
        alias_key = norm_key(alias)
        if alias_key in allowed_lookup:
            return allowed_lookup[alias_key]
        return alias

    if key in allowed_lookup:
        return allowed_lookup[key]

    # E-code exact first
    e_match = re.search(r"\be([0-9]{3,4}[a-z]{0,3})\b", key)
    if e_match:
        ecode = f"e{e_match.group(1)}"
        exact = [y for y in allowed_labels if norm_key(y).startswith(ecode)]
        if len(exact) == 1:
            return exact[0]

        # fallback to numeric-only ecode
        digits_only = re.match(r"e([0-9]{3,4})", ecode)
        if digits_only:
            ecode_base = f"e{digits_only.group(1)}"
            base_matches = [y for y in allowed_labels if norm_key(y).startswith(ecode_base)]
            if len(base_matches) == 1:
                return base_matches[0]

    prefix_matches = [y for y in allowed_labels if norm_key(y).startswith(key)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    contain_matches = [y for y in allowed_labels if key and key in norm_key(y)]
    if len(contain_matches) == 1:
        return contain_matches[0]

    return raw


def canonicalize_pred_labels(
    pred: List[str],
    allowed_lookup: Dict[str, str],
    allowed_labels: List[str]
) -> List[str]:
    out = []
    for x in pred:
        c = canonicalize_one_label(x, allowed_lookup, allowed_labels)
        if c:
            out.append(c)
    return normalize_labels(out)


def find_ecode_label(ecode: str, allowed_labels: List[str]) -> str:
    ecode = norm_key(ecode)

    exact = [lbl for lbl in allowed_labels if norm_key(lbl).startswith(ecode)]
    if len(exact) == 1:
        return exact[0]

    # fallback to numeric-only
    m = re.match(r"e([0-9]{3,4})", ecode)
    if m:
        base = f"e{m.group(1)}"
        matches = [lbl for lbl in allowed_labels if norm_key(lbl).startswith(base)]
        if len(matches) == 1:
            return matches[0]

    return ""


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
    label_to_category: Dict[str, str],
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
        for k, v in mp.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue

            hl = v.get("hazard_label", "")
            hc = v.get("hazard_category_label", "")

            if isinstance(hl, str):
                hl = hl.strip() or k.strip()
                if hl:
                    hazard_labels.add(hl)
            elif isinstance(hl, list):
                for x in hl:
                    if isinstance(x, str) and x.strip():
                        hazard_labels.add(x.strip())

            if isinstance(hc, str):
                hc = hc.strip() or k.strip()
                if hc:
                    hazard_category_labels.add(hc)
            elif isinstance(hc, list):
                for x in hc:
                    if isinstance(x, str) and x.strip():
                        hazard_category_labels.add(x.strip())

    for k, v in label_to_category.items():
        if isinstance(k, str) and k.strip():
            hazard_labels.add(k.strip())
        if isinstance(v, str) and v.strip():
            hazard_category_labels.add(v.strip())

    return sorted(hazard_labels), sorted(hazard_category_labels)


# =========================
# Mapping compilers
# =========================
def compile_ngram_mapping(mapping_dict: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    compiled = {}
    for k, v in mapping_dict.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue

        nk = normalize_text_for_match(k)
        if not nk:
            continue

        hl = v.get("hazard_label", "")
        hc = v.get("hazard_category_label", "")

        if isinstance(hl, str):
            hazard_label = [hl.strip()] if hl.strip() else []
        elif isinstance(hl, list):
            hazard_label = normalize_labels(hl)
        else:
            hazard_label = []

        if isinstance(hc, str):
            hazard_category_label = [hc.strip()] if hc.strip() else []
        elif isinstance(hc, list):
            hazard_category_label = normalize_labels(hc)
        else:
            hazard_category_label = []

        compiled[nk] = {
            "hazard_label": hazard_label,
            "hazard_category_label": hazard_category_label,
        }
    return compiled


def compile_label_to_category_map(
    label_to_category_raw: Dict[str, Any],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_category_lookup: Dict[str, str],
) -> Dict[str, str]:
    compiled = {}
    for raw_label, raw_cat in label_to_category_raw.items():
        if not isinstance(raw_label, str) or not isinstance(raw_cat, str):
            continue
        nlabel = norm_key(raw_label)
        ncat = norm_key(raw_cat)
        if not nlabel or not ncat:
            continue

        canon_label = allowed_hazard_lookup.get(nlabel, raw_label.strip())
        canon_cat = allowed_hazard_category_lookup.get(ncat, raw_cat.strip())
        compiled[canon_label] = canon_cat
    return compiled


def fill_categories_from_labels(
    labels: List[str],
    label_to_category_map: Dict[str, str]
) -> List[str]:
    out = []
    for lbl in labels:
        if lbl in label_to_category_map:
            out.append(label_to_category_map[lbl])
    return normalize_labels(out)


# =========================
# HAS HAZARDS:
# exact raw_hazard map first, then attr map
# =========================
def infer_label_from_attr(
    attr: str,
    gold_labels: List[str],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_labels: List[str]
) -> str:
    if not gold_labels:
        return ""

    nattr = norm_key(attr)
    gold_norm = {norm_key(g): g for g in gold_labels}

    if len(gold_labels) == 1:
        return gold_labels[0]

    if nattr in gold_norm:
        return gold_norm[nattr]

    if "aflatoxin" in nattr and "Aflatoxins" in gold_labels:
        return "Aflatoxins"

    for g in gold_labels:
        ng = norm_key(g)
        if ng == nattr or ng in nattr or nattr in ng:
            return g

    can_attr = canonicalize_one_label(attr, allowed_hazard_lookup, allowed_hazard_labels)
    if can_attr in gold_labels:
        return can_attr

    return ""


def infer_category_from_hazard(
    raw_hazard: str,
    raw_category: str,
    gold_categories: List[str],
    chosen_label: str,
    label_to_category_map: Dict[str, str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_category_labels: List[str],
) -> str:
    if not gold_categories:
        return ""

    nrawhaz = norm_key(raw_hazard)
    nrawcat = norm_key(raw_category)
    gold_norm = {norm_key(g): g for g in gold_categories}

    if len(gold_categories) == 1:
        return gold_categories[0]

    # full raw hazard string first
    if "undeclared" in nrawhaz and "allergen" in nrawhaz and "allergens" in gold_norm:
        return gold_norm["allergens"]

    if "residues of veterinary medicinal products" in nrawcat and "veterinary drug residues" in gold_norm:
        return gold_norm["veterinary drug residues"]

    if chosen_label and chosen_label in label_to_category_map:
        mapped = label_to_category_map[chosen_label]
        if mapped in gold_categories:
            return mapped

    if nrawcat in gold_norm:
        return gold_norm[nrawcat]

    if "pesticide residues" in nrawcat and "Pesticide residues" in gold_categories:
        return "Pesticide residues"
    if "food additives" in nrawcat and "Food additives" in gold_categories:
        return "Food additives"
    if "novel food" in nrawcat and "Novel food" in gold_categories:
        return "Novel food"

    can_cat = canonicalize_one_label(raw_category, allowed_hazard_category_lookup, allowed_hazard_category_labels)
    if can_cat in gold_categories:
        return can_cat

    return ""


def build_has_hazards_exact_maps(
    has_labelled: List[Dict[str, Any]],
    label_to_category_map: Dict[str, str],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    raw_hazard_to_label_counts = defaultdict(Counter)
    raw_hazard_to_cat_counts = defaultdict(Counter)
    attr_to_label_counts = defaultdict(Counter)
    attr_to_cat_counts = defaultdict(Counter)

    for row in has_labelled:
        gold_labels = normalize_labels(row.get("hazard_label", []))
        gold_categories = normalize_labels(row.get("hazard_category_label", []))

        for h in row.get("hazards", []):
            if not isinstance(h, dict):
                continue

            raw_hazard = h.get("hazard", "")
            raw_category = h.get("category", "")
            nrawhaz = norm_key(raw_hazard)
            attr = extract_hazard_attribute(raw_hazard)
            nattr = norm_key(attr)

            if not nrawhaz:
                continue

            chosen_label = infer_label_from_attr(
                attr=attr,
                gold_labels=gold_labels,
                allowed_hazard_lookup=allowed_hazard_lookup,
                allowed_hazard_labels=allowed_hazard_labels,
            )
            if chosen_label:
                raw_hazard_to_label_counts[nrawhaz][chosen_label] += 1
                attr_to_label_counts[nattr][chosen_label] += 1

            chosen_cat = infer_category_from_hazard(
                raw_hazard=raw_hazard,
                raw_category=raw_category,
                gold_categories=gold_categories,
                chosen_label=chosen_label,
                label_to_category_map=label_to_category_map,
                allowed_hazard_category_lookup=allowed_hazard_category_lookup,
                allowed_hazard_category_labels=allowed_hazard_category_labels,
            )
            if chosen_cat:
                raw_hazard_to_cat_counts[nrawhaz][chosen_cat] += 1
                attr_to_cat_counts[nattr][chosen_cat] += 1

    def pick_best(counter_map: Dict[str, Counter]) -> Dict[str, str]:
        out = {}
        for k, cnt in counter_map.items():
            out[k] = sorted(cnt.items(), key=lambda z: (-z[1], len(z[0]), z[0]))[0][0]
        return out

    return (
        pick_best(raw_hazard_to_label_counts),
        pick_best(raw_hazard_to_cat_counts),
        pick_best(attr_to_label_counts),
        pick_best(attr_to_cat_counts),
    )


def map_has_hazard_item(
    raw_hazard: str,
    raw_category: str,
    raw_hazard_to_label: Dict[str, str],
    raw_hazard_to_cat: Dict[str, str],
    attr_to_label: Dict[str, str],
    attr_to_cat: Dict[str, str],
    label_to_category_map: Dict[str, str],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[List[str], List[str], bool]:
    nrawhaz = norm_key(raw_hazard)
    attr = extract_hazard_attribute(raw_hazard)
    nattr = norm_key(attr)

    labels = []
    cats = []
    resolved = False

    # 1) exact full raw hazard mapping
    if nrawhaz in raw_hazard_to_label:
        labels.append(raw_hazard_to_label[nrawhaz])
        resolved = True
    elif nattr in attr_to_label:
        labels.append(attr_to_label[nattr])
        resolved = True
    else:
        label = canonicalize_one_label(attr, allowed_hazard_lookup, allowed_hazard_labels)
        if label:
            labels.append(label)

    chosen_label = labels[0] if labels else ""

    # 2) exact full raw hazard category mapping
    if nrawhaz in raw_hazard_to_cat:
        cats.append(raw_hazard_to_cat[nrawhaz])
        resolved = True
    elif nattr in attr_to_cat:
        cats.append(attr_to_cat[nattr])
        resolved = True
    else:
        nrawcat = norm_key(raw_category)

        if "undeclared" in nrawhaz and "allergen" in nrawhaz:
            cats.append("Allergens")
        elif "residues of veterinary medicinal products" in nrawcat:
            cats.append("Veterinary drug residues")
        elif chosen_label and chosen_label in label_to_category_map:
            cats.append(label_to_category_map[chosen_label])
        elif "pesticide residues" in nrawcat:
            cats.append("Pesticide residues")
        elif "food additives" in nrawcat:
            cats.append("Food additives")
        elif "novel food" in nrawcat:
            cats.append("Novel food")
        else:
            ccat = canonicalize_one_label(raw_category, allowed_hazard_category_lookup, allowed_hazard_category_labels)
            if ccat:
                cats.append(ccat)

    labels = canonicalize_pred_labels(labels, allowed_hazard_lookup, allowed_hazard_labels)
    cats = canonicalize_pred_labels(cats, allowed_hazard_category_lookup, allowed_hazard_category_labels)

    return labels, cats, resolved


# =========================
# NO HAZARDS rules
# =========================
ALLERGEN_CUES = [
    "allergen", "allergens",
    "undeclared", "not declared", "not mentioned on the label",
    "not highlighted on labels", "not highlighted on the label",
    "contains traces of", "unmentioned sulphites", "unmentioned sulfites"
]

PATHOGEN_PATTERNS = {
    "Cronobacter sakazakii": [r"\bcronobacter sakazakii\b"],
    "Rotavirus": [r"\brotavirus\b"],
    "Norovirus": [r"\bnorovirus\b"],
    "Listeria monocytogenes": [r"\blisteria monocytogenes\b"],
    "Listeria spp": [r"\blisteria\b"],
    "Bacillus cereus": [r"\bbacillus cereus\b"],
    "Escherichia coli": [r"\be\s*coli\b", r"\bescherichia coli\b"],
    "Campylobacter spp": [r"\bcampylobacter\b"],
    "Vibrio vulnificus": [r"\bvibrio vulnificus\b"],
    "Vibrio spp": [r"\bvibrio\b"],
    "Hepatitis A virus": [r"\bhepatitis a\b"],
    "Hepatitis E virus": [r"\bhepatitis e\b"],
}

OTHER_HAZARD_PATTERNS = {
    "Sibutramine": [r"\bsibutramine\b", r"\bsibutramina\b"],
    "Antibiotics": [r"\bantibiotic residue\b", r"\bantibiotic residues\b", r"\bantibiotic\b", r"\bantibiotics\b"],
    "Monacolin K": [r"\bmonacolin k\b", r"\bmonacolina k\b"],
    "Anisakis": [r"\banisakis\b", r"\banisakidae\b"],
    "Parasitic infestation": [r"\bparasitic infestation\b", r"\bparasite\b", r"\bparasitic\b"],
    "Porcine DNA": [r"\bporcine dna\b", r"\bpig species\b", r"\bsuino\b"],
    "Poultry DNA": [r"\bpoultry dna\b", r"\bchicken species\b", r"\bavicole\b"],
    "Cannabinoid": [r"\bcannabinoid", r"\bcbd\b", r"\bthc\b", r"\bthcp\b", r"\bhhc\b"],
    "Lead": [r"\blead\b"],
    "Yohimbine": [r"\byohimbine\b", r"\byohimbe\b"],
    "Ethylene oxide": [r"\bethylene oxide\b"],
}


def direct_mapping_hits(
    subject: str,
    mapping1: Dict[str, Dict[str, List[str]]],
    mapping2: Dict[str, Dict[str, List[str]]],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    tokens = tokenize(subject)
    ngrams = generate_ngrams(tokens, max_n=5)

    labels = []
    categories = []
    hits1 = []
    hits2 = []

    for ng in ngrams:
        if ng in mapping1:
            hits1.append(ng)
            labels.extend(mapping1[ng]["hazard_label"])
            categories.extend(mapping1[ng]["hazard_category_label"])

    for ng in ngrams:
        if ng in mapping2:
            hits2.append(ng)
            labels.extend(mapping2[ng]["hazard_label"])
            categories.extend(mapping2[ng]["hazard_category_label"])

    labels = canonicalize_pred_labels(labels, allowed_hazard_lookup, allowed_hazard_labels)
    categories = canonicalize_pred_labels(categories, allowed_hazard_category_lookup, allowed_hazard_category_labels)

    return labels, categories, sorted(list(set(hits1))), sorted(list(set(hits2)))


def detect_allergens_from_subject(subject: str, allowed_hazard_lookup: Dict[str, str]) -> List[str]:
    text = normalize_text_for_match(subject)
    if any(cue in text for cue in ALLERGEN_CUES):
        if "allergens" in allowed_hazard_lookup:
            return [allowed_hazard_lookup["allergens"]]
        return ["Allergens"]
    return []


def detect_salmonella_from_subject(
    subject: str,
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
) -> List[str]:
    text = norm_key(subject)

    salmonella_specific = []
    for lbl in allowed_hazard_labels:
        if not lbl.startswith("Salmonella "):
            continue
        if lbl == "Salmonella spp":
            continue
        nl = norm_key(lbl)
        if nl and nl in text:
            salmonella_specific.append(lbl)

    salmonella_specific = normalize_labels(salmonella_specific)
    if salmonella_specific:
        return salmonella_specific

    if "salmonella" in text:
        if "salmonella spp" in allowed_hazard_lookup:
            return [allowed_hazard_lookup["salmonella spp"]]
        return ["Salmonella spp"]

    return []


def detect_pathogens_from_subject(subject: str, allowed_hazard_lookup: Dict[str, str]) -> List[str]:
    text = normalize_text_for_match(subject)
    labels = []
    for lbl, patterns in PATHOGEN_PATTERNS.items():
        if regex_any(patterns, text):
            key = norm_key(lbl)
            labels.append(allowed_hazard_lookup.get(key, lbl))
    return normalize_labels(labels)


def detect_additives_from_subject(
    subject: str,
    allowed_hazard_labels: List[str],
    allowed_hazard_lookup: Dict[str, str],
) -> List[str]:
    labels = []

    ecodes = extract_ecodes(subject)
    for e in ecodes:
        lbl = find_ecode_label(e, allowed_hazard_labels)
        if lbl:
            labels.append(lbl)

    text = normalize_text_for_match(subject)

    if (
        "unauthorised food additive" in text or
        "unauthorized food additive" in text or
        "unauthorised food additives" in text or
        "unauthorized food additives" in text or
        "unauthorised additive" in text or
        "unauthorized additive" in text or
        "unauthorised additives" in text or
        "unauthorized additives" in text
    ):
        if not labels:
            if "food additives" in allowed_hazard_lookup:
                labels.append(allowed_hazard_lookup["food additives"])
            else:
                labels.append("Food additives")

    return normalize_labels(labels)


def detect_novel_food_from_subject(subject: str, allowed_hazard_lookup: Dict[str, str]) -> List[str]:
    text = normalize_text_for_match(subject)
    labels = []

    if "nicotinamide mononucleotide" in text or re.search(r"\bnmn\b", text):
        labels.append(allowed_hazard_lookup.get("nicotinamide mononucleotide", "Nicotinamide mononucleotide"))
        return normalize_labels(labels)

    if "novel food ingredient" in text:
        labels.append(allowed_hazard_lookup.get("novel food ingredient", "Novel food ingredient"))
        return normalize_labels(labels)

    if "novel food" in text or "unauthorised novel food" in text or "unauthorized novel food" in text:
        labels.append(allowed_hazard_lookup.get("novel food", "Novel food"))
        return normalize_labels(labels)

    return []


def detect_pesticides_from_subject(
    subject: str,
    pesticide_labels: List[str],
    allowed_hazard_lookup: Dict[str, str],
) -> List[str]:
    text = normalize_text_for_match(subject)
    found = []

    if "residues of pesticides" in text or "pesticide residues" in text:
        if "pesticide residues" in allowed_hazard_lookup:
            found.append(allowed_hazard_lookup["pesticide residues"])
        else:
            found.append("Pesticide residues")

    for lbl in pesticide_labels:
        nl = norm_key(lbl)
        if nl and nl in norm_key(text):
            found.append(lbl)

    if "chlorpyrifos ethyl" in text or "chlorpyriphos ethyl" in text:
        found.append(allowed_hazard_lookup.get("chlorpyriphos-ethyl", "Chlorpyriphos-ethyl"))

    return normalize_labels(found)


def detect_foreign_bodies_from_subject(subject: str, allowed_hazard_lookup: Dict[str, str]) -> List[str]:
    text = normalize_text_for_match(subject)

    cues = [
        "foreign body",
        "foreign bodies",
        "glass fragment",
        "glass fragments",
        "piece of glass",
        "pieces of glass",
        "plastic fragment",
        "plastic fragments",
        "plastic particle",
        "plastic particles",
        "piece of plastic",
        "pieces of plastic",
        "metal fragment",
        "metal fragments",
        "metal particles",
        "piece of wood",
        "pieces of wood",
        "dead insects",
        "bean weevil",
        "nail",
    ]

    if any(c in text for c in cues):
        if "foreign bodies" in allowed_hazard_lookup:
            return [allowed_hazard_lookup["foreign bodies"]]
        return ["Foreign bodies"]

    return []


def detect_vitamins_from_subject(subject: str, allowed_hazard_lookup: Dict[str, str]) -> List[str]:
    text = normalize_text_for_match(subject)
    labels = []

    if "vitamin b6" in text:
        labels.append(allowed_hazard_lookup.get("vitamin b6", "Vitamin B6"))

    if "niacin" in text or "nicotinic acid" in text or "vitamin b3" in text:
        labels.append(allowed_hazard_lookup.get("vitamin b3", "Vitamin B3"))

    if not labels and "vitamin content" in text:
        labels.append(allowed_hazard_lookup.get("vitamins", "Vitamins"))

    return normalize_labels(labels)


def detect_other_hazards_from_subject(subject: str, allowed_hazard_lookup: Dict[str, str]) -> List[str]:
    text = normalize_text_for_match(subject)
    labels = []

    for lbl, patterns in OTHER_HAZARD_PATTERNS.items():
        if regex_any(patterns, text):
            key = norm_key(lbl)
            labels.append(allowed_hazard_lookup.get(key, lbl))

    return normalize_labels(labels)


def detect_direct_hazard_labels_from_subject(
    subject: str,
    allowed_hazard_lookup: Dict[str, str]
) -> List[str]:
    text = normalize_text_for_match(subject)
    labels = []

    pattern_to_label = {
        r"\bofficial certificates?\b": "Documentation",
        r"\boriginal certificates?\b": "Documentation",
        r"\babsence of original certificates?\b": "Documentation",
        r"\babsence of official certificates?\b": "Documentation",
        r"\babsence of health certificate\b": "Documentation",
        r"\bhealth certificate\b": "Documentation",
        r"\bcommon health entry document\b": "Documentation",
        r"\bched\b": "Documentation",
        r"\bchedp\b": "Documentation",

        r"\blabelling deficiencies\b": "Labelling",
        r"\blabeling deficiencies\b": "Labelling",
        r"\babsence of labelling\b": "Labelling",
        r"\babsence of labeling\b": "Labelling",
        r"\bincorrect labeling\b": "Labelling",
        r"\bincorrect labelling\b": "Labelling",
        r"\bmissing on the labelling\b": "Labelling",
        r"\bmissing on the labeling\b": "Labelling",

        r"\borganoleptic defect\b": "Organoleptic aspects",
        r"\borganoleptic changes\b": "Organoleptic aspects",
        r"\borganoleptic non conformity\b": "Organoleptic aspects",
        r"\babnormal smell\b": "Organoleptic aspects",
        r"\bodour\b": "Organoleptic aspects",
        r"\bodor\b": "Organoleptic aspects",

        r"\bmould\b": "Mould",
        r"\bmoulds\b": "Mould",
        r"\bmold\b": "Mould",
        r"\bmolds\b": "Mould",

        r"\bbamboo\b": "Bamboo",
        r"\bmelamine\b": "Melamine",
        r"\bethylene oxide\b": "Ethylene oxide",
        r"\byohimbine\b": "Yohimbine",
        r"\byohimbe\b": "Yohimbine",
        r"\blead\b": "Lead",
        r"\bformaldehyde\b": "Formaldehyde",
        r"\bveterinary checks?\b": "Veterinary control",
        r"\bveterinary control\b": "Veterinary control",
        r"\bfeed additives?\b": "Feed additives",
        r"\bfeed additive\b": "Feed additives",
        r"\bpoor temperature control\b": "Temperature control",
        r"\bcold chain\b": "Temperature control",
    }

    for pattern, label in pattern_to_label.items():
        if re.search(pattern, text):
            key = norm_key(label)
            labels.append(allowed_hazard_lookup.get(key, label))

    return normalize_labels(labels)


def remove_generic_when_specific_exists(labels: List[str], generic_label: str, prefix: str) -> List[str]:
    has_specific = any(x.startswith(prefix) and x != generic_label for x in labels)
    if has_specific and generic_label in labels:
        labels = [x for x in labels if x != generic_label]
    return labels


def infer_no_hazards(
    item: Dict[str, Any],
    mapping1: Dict[str, Dict[str, List[str]]],
    mapping2: Dict[str, Dict[str, List[str]]],
    label_to_category_map: Dict[str, str],
    pesticide_labels: List[str],
    allowed_hazard_lookup: Dict[str, str],
    allowed_hazard_labels: List[str],
    allowed_hazard_category_lookup: Dict[str, str],
    allowed_hazard_category_labels: List[str],
) -> Tuple[List[str], List[str], List[str], str]:
    subject = item.get("subject", "")
    text = normalize_text_for_match(subject)

    labels, categories, hits1, hits2 = direct_mapping_hits(
        subject=subject,
        mapping1=mapping1,
        mapping2=mapping2,
        allowed_hazard_lookup=allowed_hazard_lookup,
        allowed_hazard_labels=allowed_hazard_labels,
        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
        allowed_hazard_category_labels=allowed_hazard_category_labels,
    )

    direct_labels = []
    direct_labels.extend(detect_direct_hazard_labels_from_subject(subject, allowed_hazard_lookup))
    direct_labels.extend(detect_additives_from_subject(subject, allowed_hazard_labels, allowed_hazard_lookup))
    direct_labels.extend(detect_novel_food_from_subject(subject, allowed_hazard_lookup))
    direct_labels.extend(detect_pesticides_from_subject(subject, pesticide_labels, allowed_hazard_lookup))
    direct_labels.extend(detect_foreign_bodies_from_subject(subject, allowed_hazard_lookup))
    direct_labels.extend(detect_allergens_from_subject(subject, allowed_hazard_lookup))
    direct_labels.extend(detect_salmonella_from_subject(subject, allowed_hazard_lookup, allowed_hazard_labels))
    direct_labels.extend(detect_pathogens_from_subject(subject, allowed_hazard_lookup))
    direct_labels.extend(detect_other_hazards_from_subject(subject, allowed_hazard_lookup))
    direct_labels.extend(detect_vitamins_from_subject(subject, allowed_hazard_lookup))

    labels = normalize_labels(labels + direct_labels)
    labels = canonicalize_pred_labels(labels, allowed_hazard_lookup, allowed_hazard_labels)

    # ========= cleanup on labels =========
    labels = remove_generic_when_specific_exists(labels, "Salmonella spp", "Salmonella ")
    labels = remove_generic_when_specific_exists(labels, "Listeria spp", "Listeria ")
    labels = remove_generic_when_specific_exists(labels, "Vibrio spp", "Vibrio ")
    labels = remove_generic_when_specific_exists(labels, "Cronobacter spp", "Cronobacter ")

    # generic allergen
    if "Allergens" in labels and "Labelling" not in labels:
        labels = ["Allergens"]

    # exact E-codes: keep generic Food additives only if there is no specific additive label
    specific_ecode_present = any(re.match(r"^E[0-9]{3,4}", x) for x in labels)
    if specific_ecode_present and "Food additives" in labels:
        labels = [x for x in labels if x != "Food additives"]

    # formaldehyde + unauthorized food additives
    if "formaldehyde" in text and (
        "unauthorised food additive" in text or
        "unauthorized food additive" in text or
        "unauthorised food additives" in text or
        "unauthorized food additives" in text
    ):
        if "Food additives" not in labels:
            labels.append("Food additives")

    # novel food cleanup
    generic_novel = {"Novel food", "Novel food ingredient"}
    specific_novel_like = [x for x in labels if x not in generic_novel and (
        x == "Cannabinoid" or
        x in {"Clitoria ternatea", "Tongkat ali (Eurycoma longifolia)", "Nicotinamide mononucleotide"}
    )]

    if labels == ["Novel food ingredient"]:
        labels = ["Novel food"]

    if "Cannabinoid" in labels and "Novel food ingredient" in labels:
        labels = [x for x in labels if x != "Novel food ingredient"]

    if specific_novel_like:
        labels = [x for x in labels if x not in {"Novel food", "Novel food ingredient"}]

    # bamboo novel food: keep Novel food only
    if "novel food" in text and "Bamboo" in labels and "plastic" not in text and "tableware" not in text:
        labels = [x for x in labels if x != "Bamboo"]

    # veterinary/feed administrative labels
    if "veterinary checks" in text or "veterinary control" in text:
        if "Veterinary control" not in labels:
            labels.append("Veterinary control")

    if "feed additive" in text or "feed additives" in text:
        if "Feed additives" not in labels:
            labels.append("Feed additives")

    # anisakis
    if "anisakis" in text and "Anisakis" not in labels:
        labels.append("Anisakis")

    labels = normalize_labels(labels)
    labels = canonicalize_pred_labels(labels, allowed_hazard_lookup, allowed_hazard_labels)

    # ========= categories =========
    categories = normalize_labels(categories + fill_categories_from_labels(labels, label_to_category_map))

    # direct category fixes
    if "Ethylene oxide" in labels:
        if "Pesticide residues" not in categories:
            categories.append("Pesticide residues")

    if "Cannabinoid" in labels and (
        "novel food" in text or
        "novel food ingredient" in text or
        "cbd" in text or
        "thc" in text or
        "hhc" in text
    ):
        categories = [x for x in categories if x != "Cannabinoid"]
        if "Novel food" not in categories:
            categories.append("Novel food")

    if "Anisakis" in labels:
        if "Parasitic infestation" not in categories:
            categories.append("Parasitic infestation")
        categories = [x for x in categories if x != "Anisakis"]

    if "Veterinary control" in labels and "Veterinary control" not in categories:
        categories.append("Veterinary control")

    if "Feed additives" in labels and "Feed additives" not in categories:
        categories.append("Feed additives")

    categories = normalize_labels(categories)
    categories = canonicalize_pred_labels(categories, allowed_hazard_category_lookup, allowed_hazard_category_labels)

    source = "rules_only"
    if hits1 and hits2:
        source = "mapping1+mapping2+rules"
    elif hits1:
        source = "mapping1+rules"
    elif hits2:
        source = "mapping2+rules"

    return labels, categories, sorted(list(set(hits1 + hits2))), source


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    has_data = load_json(HAS_HAZARDS_LABELLED_PATH)
    no_data = load_json(NO_HAZARDS_LABELLED_PATH)

    has_hazard_label_inventory = load_json(HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH)
    has_hazard_category_inventory = load_json(HAS_HAZARDS_MAPPING_HAZARD_CATEGORY_LABEL_PATH)
    no_mapping1_raw = load_json(NO_HAZARDS_MAPPING1_PATH)
    no_mapping2_raw = load_json(NO_HAZARDS_MAPPING2_PATH)
    label_to_category_raw = load_json(MAPPING_LABEL_TO_CATEGORY_PATH)

    allowed_hazard_labels, allowed_hazard_category_labels = build_allowed_label_sets(
        has_labelled=has_data,
        no_labelled=no_data,
        has_hazard_label_inventory=has_hazard_label_inventory,
        has_hazard_category_inventory=has_hazard_category_inventory,
        no_mapping1=no_mapping1_raw,
        no_mapping2=no_mapping2_raw,
        label_to_category=label_to_category_raw,
    )

    # ===== preferred canonical forms from gold data =====
    preferred_hazard_lookup = build_preferred_lookup_from_gold(has_data + no_data, "hazard_label")
    preferred_hazard_category_lookup = build_preferred_lookup_from_gold(has_data + no_data, "hazard_category_label")

    allowed_hazard_lookup = merge_preferred_and_allowed(preferred_hazard_lookup, allowed_hazard_labels)
    allowed_hazard_category_lookup = merge_preferred_and_allowed(preferred_hazard_category_lookup, allowed_hazard_category_labels)

    allowed_hazard_set = set(allowed_hazard_labels)
    allowed_hazard_category_set = set(allowed_hazard_category_labels)

    label_to_category_map = compile_label_to_category_map(
        label_to_category_raw,
        allowed_hazard_lookup=allowed_hazard_lookup,
        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
    )

    (
        raw_hazard_to_label,
        raw_hazard_to_cat,
        attr_to_label,
        attr_to_cat,
    ) = build_has_hazards_exact_maps(
        has_labelled=has_data,
        label_to_category_map=label_to_category_map,
        allowed_hazard_lookup=allowed_hazard_lookup,
        allowed_hazard_labels=allowed_hazard_labels,
        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
        allowed_hazard_category_labels=allowed_hazard_category_labels,
    )

    mapping1 = compile_ngram_mapping(no_mapping1_raw)
    mapping2 = compile_ngram_mapping(no_mapping2_raw)

    pesticide_labels = sorted(
        [lbl for lbl, cat in label_to_category_map.items() if cat == "Pesticide residues"],
        key=lambda x: len(normalize_text_for_match(x)),
        reverse=True
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
        "has_hazards_records": 0,
        "has_hazards_exact_raw_hazard_hits": 0,
        "has_hazards_attr_fallback_hits": 0,
        "no_hazards_records": 0,
        "no_hazards_mapping1_hits": 0,
        "no_hazards_mapping2_hits": 0,
        "no_hazards_empty_after_rules": 0,
    }

    for idx, item in enumerate(tqdm(eval_pool, desc="Evaluating hazard rule-only v6")):
        time.sleep(0.003)
        group = item["_group"]

        try:
            subject = item.get("subject", "")
            hazards = item.get("hazards", [])

            if group == "has_hazards":
                source_stats["has_hazards_records"] += 1

                pred_hazard_labels = []
                pred_hazard_categories = []
                matched_keys = []

                for h in hazards:
                    if not isinstance(h, dict):
                        continue

                    raw_hazard = h.get("hazard", "")
                    raw_category = h.get("category", "")

                    labels, cats, _ = map_has_hazard_item(
                        raw_hazard=raw_hazard,
                        raw_category=raw_category,
                        raw_hazard_to_label=raw_hazard_to_label,
                        raw_hazard_to_cat=raw_hazard_to_cat,
                        attr_to_label=attr_to_label,
                        attr_to_cat=attr_to_cat,
                        label_to_category_map=label_to_category_map,
                        allowed_hazard_lookup=allowed_hazard_lookup,
                        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
                        allowed_hazard_labels=allowed_hazard_labels,
                        allowed_hazard_category_labels=allowed_hazard_category_labels,
                    )

                    pred_hazard_labels.extend(labels)
                    pred_hazard_categories.extend(cats)

                    nrawhaz = norm_key(raw_hazard)
                    nattr = norm_key(extract_hazard_attribute(raw_hazard))
                    matched_keys.append(nrawhaz)

                    if nrawhaz in raw_hazard_to_label or nrawhaz in raw_hazard_to_cat:
                        source_stats["has_hazards_exact_raw_hazard_hits"] += 1
                    elif nattr in attr_to_label or nattr in attr_to_cat:
                        source_stats["has_hazards_attr_fallback_hits"] += 1

                prediction_source = "has_hazards_exact_raw_hazard_mapping"

                pred_hazard_labels = canonicalize_pred_labels(
                    pred_hazard_labels, allowed_hazard_lookup, allowed_hazard_labels
                )
                pred_hazard_categories = canonicalize_pred_labels(
                    pred_hazard_categories, allowed_hazard_category_lookup, allowed_hazard_category_labels
                )

            else:
                source_stats["no_hazards_records"] += 1

                pred_hazard_labels, pred_hazard_categories, matched_keys, prediction_source = infer_no_hazards(
                    item=item,
                    mapping1=mapping1,
                    mapping2=mapping2,
                    label_to_category_map=label_to_category_map,
                    pesticide_labels=pesticide_labels,
                    allowed_hazard_lookup=allowed_hazard_lookup,
                    allowed_hazard_labels=allowed_hazard_labels,
                    allowed_hazard_category_lookup=allowed_hazard_category_lookup,
                    allowed_hazard_category_labels=allowed_hazard_category_labels,
                )

                if any(k in mapping1 for k in matched_keys):
                    source_stats["no_hazards_mapping1_hits"] += 1
                if any(k in mapping2 for k in matched_keys):
                    source_stats["no_hazards_mapping2_hits"] += 1
                if not pred_hazard_labels and not pred_hazard_categories:
                    source_stats["no_hazards_empty_after_rules"] += 1

            pred_hazard_labels = [x for x in pred_hazard_labels if x in allowed_hazard_set]
            pred_hazard_categories = [x for x in pred_hazard_categories if x in allowed_hazard_category_set]

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
            out_item["matched_mapping_keys"] = sorted(list(set(matched_keys)))
            results.append(out_item)

            if joint_match == 0:
                wrong_cases.append(out_item)

            if idx < 5:
                print("\n[Sample Prediction]")
                print(json.dumps({
                    "group": group,
                    "reference": item.get("reference", ""),
                    "subject": subject,
                    "gold_hazard_label": gold_hazard_label,
                    "pred_hazard_label": pred_hazard_labels,
                    "gold_hazard_category_label": gold_hazard_category_label,
                    "pred_hazard_category_label": pred_hazard_categories,
                    "joint_match": joint_match,
                    "prediction_source": prediction_source
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
        "use_prompt": False,
        "has_hazards_use_exact_raw_hazard_mapping_first": True,
        "use_mapping_hazard_label_to_hazard_category_label_as_helper_only": True,
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