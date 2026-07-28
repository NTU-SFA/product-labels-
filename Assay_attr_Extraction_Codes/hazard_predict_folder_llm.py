"""
混合 hazard 抽取（folder 版，单文件自包含，不 import 其它 .py）：
  - has_hazards 记录（有结构化 hazards 字段）-> table-lookup 引擎（本文件内联，可靠）
  - no_hazards  记录（只有 subject）        -> Claude LLM（泛化更好）

LLM 候选标签 = mapping_hazard_label_to_hazard_category_label.json 的 keys（937 个）。
hazard_category_label 一律由该映射从 hazard_label 推导。
两段输出都过 canon937 统一规范化到 937 词表（别名/过敏原表见 hazard_canon_config.json）。

预测写进已有的 hazard_label / hazard_category_label 字段，保留 product_label，
输出到新的 *_hazard_llm 文件夹。

外部依赖（数据/Prompt，非 .py）：
  - hazard/ 下的 gold 与 mapping json
  - hazard_no_hazards_prompt_V1.txt
  - hazard_canon_config.json
Python 包：langchain_aws, langchain_core, tqdm
环境变量：AWS_BEARER_TOKEN_BEDROCK（下方有 setdefault 兜底）
"""
import os
import re
import json
import glob
import unicodedata
from string import Formatter
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import PromptTemplate


# =========================
# 路径 & 数据文件
# =========================
# 项目根目录：本机默认绝对路径；Docker 里用 RASFF_ROOT=/app 覆盖，代码零改动。
RASFF_ROOT = os.environ.get("RASFF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(RASFF_ROOT, "Assay_attr_Extraction_Codes")
HAZARD_DIR = os.path.join(RASFF_ROOT, "hazard")

HAS_HAZARDS_LABELLED_PATH = os.path.join(HAZARD_DIR, "rasff_data_2020_to_2026_has_hazards_with_labels.json")
NO_HAZARDS_LABELLED_PATH = os.path.join(HAZARD_DIR, "rasff_data_2020_to_2026_no_hazards_with_labels.json")
HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH = os.path.join(HAZARD_DIR, "has_hazards_mapping_hazard_label.json")
HAS_HAZARDS_MAPPING_HAZARD_CATEGORY_LABEL_PATH = os.path.join(HAZARD_DIR, "has_hazards_mapping_hazard_category_label.json")
NO_HAZARDS_MAPPING1_PATH = os.path.join(HAZARD_DIR, "no_hazards_mapping1.json")
NO_HAZARDS_MAPPING2_PATH = os.path.join(HAZARD_DIR, "no_hazards_mapping2.json")
MAPPING_LABEL_TO_CATEGORY_PATH = os.path.join(HAZARD_DIR, "mapping_hazard_label_to_hazard_category_label.json")

# 候选 hazard_label 集 = label->category 映射文件的 keys（937 个规范标签）
LABEL_TO_CATEGORY_PATH = MAPPING_LABEL_TO_CATEGORY_PATH

# no_hazards 段用的 prompt（外部文件）
NO_HAZARD_PROMPT_PATH = os.path.join(CODE_DIR, "hazard_no_hazards_prompt_V1.txt")

# canon937 别名/过敏原配置（外部文件）
CANON_CONFIG_PATH = os.path.join(CODE_DIR, "hazard_canon_config.json")

# 待处理文件夹（product 输出）。可用 INPUT_DIR 环境变量覆盖（单个文件夹）。
INPUT_DIRS = [
    os.environ.get(
        "INPUT_DIR",
        os.path.join(CODE_DIR, "rasff_information_extraction_2024_V2_product"),
    )
]


def output_dir_for(input_dir: str) -> str:
    base = os.path.basename(input_dir.rstrip("/"))
    if base.endswith("_product"):
        base = base[: -len("_product")] + "_hazard_llm"
    else:
        base = base + "_hazard_llm"
    return os.path.join(CODE_DIR, base)


# =========================
# 运行配置
# =========================
OVERWRITE_IN_PLACE = False
SKIP_EXISTING = True          # 默认开断点续跑（LLM 调用贵，避免重跑）
DEBUG_N = None                # 每个文件只处理前 N 条（调试）；None=全量
MAX_WORKERS = 10              # no_hazards 段 LLM 调用并发数

EXCLUDE_SUFFIXES = ("_prediction_errors.json", "_hazard_prediction_errors.json")
EXCLUDE_NAMES = ("prediction_summary_folder.json", "hazard_prediction_summary_folder.json")


# =========================
# Bedrock / LLM 配置 + helper（内联自原 Claude_predict_folder.py）
# =========================
os.environ.setdefault(
    "AWS_BEARER_TOKEN_BEDROCK",
    "REPLACE_WITH_YOUR_AWS_BEARER_TOKEN_BEDROCK",
)
AWS_REGION = "ap-southeast-1"
CLAUDE_MODEL = "global.anthropic.claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 1024


def load_json(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(fp: str) -> str:
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


def safe_get(item: Dict[str, Any], key: str) -> str:
    value = item.get(key, "")
    if value is None:
        return ""
    return str(value)


def parse_json_output(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        idx = text.find("{", idx + 1)
    raise ValueError(f"Cannot parse JSON output: {text}")


def extract_template_variables(template_text: str) -> List[str]:
    vars_found = []
    for _, field_name, _, _ in Formatter().parse(template_text):
        if field_name:
            vars_found.append(field_name)
    return sorted(list(set(vars_found)))


def build_chain(prompt_path: str):
    prompt_text = load_text(prompt_path)
    prompt_variables = extract_template_variables(prompt_text)
    prompt = PromptTemplate(input_variables=prompt_variables, template=prompt_text)
    llm = ChatBedrockConverse(
        model=CLAUDE_MODEL,
        region_name=AWS_REGION,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return prompt | llm, prompt_variables


# =========================
# 规范化配置（全部外置到 hazard_canon_config.json，改表只需改文件）
#   aliases             -> canon937 别名表（变体 -> 937 标准标签）
#   allergen_foods      -> 过敏原食物名（出现即折叠为 Allergens）
#   rule_manual_aliases -> table-lookup canonicalize_one_label 的别名表
# =========================
def _load_canon_config() -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    try:
        cfg = load_json(CANON_CONFIG_PATH)
    except FileNotFoundError:
        print(f"[Warn] canon config not found, normalization runs without aliases: {CANON_CONFIG_PATH}")
        return {}, [], {}
    return (
        cfg.get("aliases", {}) or {},
        cfg.get("allergen_foods", []) or [],
        cfg.get("rule_manual_aliases", {}) or {},
    )


_ALIASES_RAW, _ALLERGEN_FOODS_RAW, _MANUAL_ALIASES = _load_canon_config()


# =========================
# table-lookup 引擎（has_hazards 段，内联自原 hazard_eval.py，仅保留 has 路径所需）
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


def extract_hazard_attribute(raw_hazard: str) -> str:
    if not isinstance(raw_hazard, str):
        return ""
    raw_hazard = raw_hazard.strip()
    if not raw_hazard:
        return ""
    return raw_hazard.split(" - ")[0].strip()


def build_preferred_lookup_from_gold(rows: List[Dict[str, Any]], field: str) -> Dict[str, str]:
    counts = defaultdict(Counter)
    for row in rows:
        for x in row.get(field, []):
            if isinstance(x, str) and x.strip():
                counts[norm_key(x)][x.strip()] += 1
    preferred = {}
    for k, c in counts.items():
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


def canonicalize_one_label(x: str, allowed_lookup: Dict[str, str], allowed_labels: List[str]) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""
    raw = x.strip()
    key = norm_key(raw)

    # 别名表外置到 hazard_canon_config.json 的 "rule_manual_aliases"
    if key in _MANUAL_ALIASES:
        alias = _MANUAL_ALIASES[key]
        alias_key = norm_key(alias)
        if alias_key in allowed_lookup:
            return allowed_lookup[alias_key]
        return alias

    if key in allowed_lookup:
        return allowed_lookup[key]

    e_match = re.search(r"\be([0-9]{3,4}[a-z]{0,3})\b", key)
    if e_match:
        ecode = f"e{e_match.group(1)}"
        exact = [y for y in allowed_labels if norm_key(y).startswith(ecode)]
        if len(exact) == 1:
            return exact[0]
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


def canonicalize_pred_labels(pred: List[str], allowed_lookup: Dict[str, str], allowed_labels: List[str]) -> List[str]:
    out = []
    for x in pred:
        c = canonicalize_one_label(x, allowed_lookup, allowed_labels)
        if c:
            out.append(c)
    return normalize_labels(out)


def build_allowed_label_sets(
    has_labelled, no_labelled, has_hazard_label_inventory, has_hazard_category_inventory,
    no_mapping1, no_mapping2, label_to_category,
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


def compile_label_to_category_map(label_to_category_raw, allowed_hazard_lookup, allowed_hazard_category_lookup) -> Dict[str, str]:
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


def infer_label_from_attr(attr, gold_labels, allowed_hazard_lookup, allowed_hazard_labels) -> str:
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


def infer_category_from_hazard(raw_hazard, raw_category, gold_categories, chosen_label,
                               label_to_category_map, allowed_hazard_category_lookup, allowed_hazard_category_labels) -> str:
    if not gold_categories:
        return ""
    nrawhaz = norm_key(raw_hazard)
    nrawcat = norm_key(raw_category)
    gold_norm = {norm_key(g): g for g in gold_categories}
    if len(gold_categories) == 1:
        return gold_categories[0]
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


def build_has_hazard_label_map(inventory: Dict[str, Any]) -> Dict[str, str]:
    """权威映射表 has_hazards_mapping_hazard_label.json -> {规范化键: hazard_label}。

    文件里两种写法：value 非空 = 改写（Fragments glass -> Foreign bodies）；
    value 为空 = 恒等（标签即 key 本身，如 Cyanide / Salmonella Altona）。
    这是 has_hazards 段 hazard_label 的第一优先来源，命中即采信。
    """
    out: Dict[str, str] = {}
    for k, v in inventory.items():
        if not isinstance(k, str) or not k.strip():
            continue
        label = v.strip() if isinstance(v, str) and v.strip() else k.strip()
        out.setdefault(norm_key(k), label)
    return out


def build_has_hazards_exact_maps(has_labelled, label_to_category_map, allowed_hazard_lookup, allowed_hazard_labels,
                                 allowed_hazard_category_lookup, allowed_hazard_category_labels):
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
            raw_category = h.get("category", "") or h.get("hazard_category", "")
            nrawhaz = norm_key(raw_hazard)
            attr = extract_hazard_attribute(raw_hazard)
            nattr = norm_key(attr)
            if not nrawhaz:
                continue
            chosen_label = infer_label_from_attr(attr, gold_labels, allowed_hazard_lookup, allowed_hazard_labels)
            if chosen_label:
                raw_hazard_to_label_counts[nrawhaz][chosen_label] += 1
                attr_to_label_counts[nattr][chosen_label] += 1
            chosen_cat = infer_category_from_hazard(
                raw_hazard, raw_category, gold_categories, chosen_label,
                label_to_category_map, allowed_hazard_category_lookup, allowed_hazard_category_labels,
            )
            if chosen_cat:
                raw_hazard_to_cat_counts[nrawhaz][chosen_cat] += 1
                attr_to_cat_counts[nattr][chosen_cat] += 1

    def pick_best(counter_map):
        out = {}
        for k, cnt in counter_map.items():
            out[k] = sorted(cnt.items(), key=lambda z: (-z[1], len(z[0]), z[0]))[0][0]
        return out

    return (pick_best(raw_hazard_to_label_counts), pick_best(raw_hazard_to_cat_counts),
            pick_best(attr_to_label_counts), pick_best(attr_to_cat_counts))


def map_has_hazard_item(raw_hazard, raw_category, raw_hazard_to_label, raw_hazard_to_cat, attr_to_label, attr_to_cat,
                        label_to_category_map, allowed_hazard_lookup, allowed_hazard_category_lookup,
                        allowed_hazard_labels, allowed_hazard_category_labels,
                        file_label_map=None):
    nrawhaz = norm_key(raw_hazard)
    attr = extract_hazard_attribute(raw_hazard)
    nattr = norm_key(attr)
    labels = []
    cats = []
    source = "fallback"

    # ① 权威映射文件优先：先按属性段查，再按 hazard 整串查。命中即采信，
    #    不再走历史语料统计表，也不再过 canon 兜底（文件给的就是规范标签）。
    file_label = ""
    if file_label_map:
        file_label = file_label_map.get(nattr) or file_label_map.get(nrawhaz) or ""

    if file_label:
        labels.append(file_label)
        source = "mapping_file"
    elif nrawhaz in raw_hazard_to_label:
        labels.append(raw_hazard_to_label[nrawhaz])
        source = "exact_raw_hazard"
    elif nattr in attr_to_label:
        labels.append(attr_to_label[nattr])
        source = "attribute_mapping"
    else:
        label = canonicalize_one_label(attr, allowed_hazard_lookup, allowed_hazard_labels)
        if label:
            labels.append(label)
        source = "canonical_fallback"

    chosen_label = labels[0] if labels else ""

    if nrawhaz in raw_hazard_to_cat:
        cats.append(raw_hazard_to_cat[nrawhaz])
    elif nattr in attr_to_cat:
        cats.append(attr_to_cat[nattr])
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

    if source != "mapping_file":                 # 映射文件的标签是权威写法，不再二次规范化
        labels = canonicalize_pred_labels(labels, allowed_hazard_lookup, allowed_hazard_labels)
    cats = canonicalize_pred_labels(cats, allowed_hazard_category_lookup, allowed_hazard_category_labels)
    return labels, cats, source


def remove_generic_when_specific_exists(labels: List[str], generic_label: str, prefix: str) -> List[str]:
    has_specific = any(x.startswith(prefix) and x != generic_label for x in labels)
    if has_specific and generic_label in labels:
        labels = [x for x in labels if x != generic_label]
    return labels


def build_context() -> Dict[str, Any]:
    """构建 has_hazards table-lookup 引擎所需的全部查表（一次性）。"""
    has_data = load_json(HAS_HAZARDS_LABELLED_PATH)
    no_data = load_json(NO_HAZARDS_LABELLED_PATH)
    has_hazard_label_inventory = load_json(HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH)
    has_hazard_category_inventory = load_json(HAS_HAZARDS_MAPPING_HAZARD_CATEGORY_LABEL_PATH)
    no_mapping1_raw = load_json(NO_HAZARDS_MAPPING1_PATH)
    no_mapping2_raw = load_json(NO_HAZARDS_MAPPING2_PATH)
    label_to_category_raw = load_json(MAPPING_LABEL_TO_CATEGORY_PATH)

    allowed_hazard_labels, allowed_hazard_category_labels = build_allowed_label_sets(
        has_data, no_data, has_hazard_label_inventory, has_hazard_category_inventory,
        no_mapping1_raw, no_mapping2_raw, label_to_category_raw,
    )
    preferred_hazard_lookup = build_preferred_lookup_from_gold(has_data + no_data, "hazard_label")
    preferred_hazard_category_lookup = build_preferred_lookup_from_gold(has_data + no_data, "hazard_category_label")
    allowed_hazard_lookup = merge_preferred_and_allowed(preferred_hazard_lookup, allowed_hazard_labels)
    allowed_hazard_category_lookup = merge_preferred_and_allowed(preferred_hazard_category_lookup, allowed_hazard_category_labels)
    label_to_category_map = compile_label_to_category_map(label_to_category_raw, allowed_hazard_lookup, allowed_hazard_category_lookup)
    raw_hazard_to_label, raw_hazard_to_cat, attr_to_label, attr_to_cat = build_has_hazards_exact_maps(
        has_data, label_to_category_map, allowed_hazard_lookup, allowed_hazard_labels,
        allowed_hazard_category_lookup, allowed_hazard_category_labels,
    )
    return {
        "file_label_map": build_has_hazard_label_map(has_hazard_label_inventory),
        "raw_hazard_to_label": raw_hazard_to_label, "raw_hazard_to_cat": raw_hazard_to_cat,
        "attr_to_label": attr_to_label, "attr_to_cat": attr_to_cat,
        "label_to_category_map": label_to_category_map,
        "allowed_hazard_labels": allowed_hazard_labels, "allowed_hazard_category_labels": allowed_hazard_category_labels,
        "allowed_hazard_lookup": allowed_hazard_lookup, "allowed_hazard_category_lookup": allowed_hazard_category_lookup,
    }


def rule_predict_has_hazards(item: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
    """has_hazards 段：用 table-lookup 引擎逐条 hazards -> hazard_label 列表。"""
    pred_labels = []
    file_sourced = set()
    for h in item.get("hazards", []):
        if not isinstance(h, dict):
            continue
        labels, _cats, _src = map_has_hazard_item(
            raw_hazard=h.get("hazard", ""), raw_category=h.get("category", "") or h.get("hazard_category", ""),
            raw_hazard_to_label=ctx["raw_hazard_to_label"], raw_hazard_to_cat=ctx["raw_hazard_to_cat"],
            attr_to_label=ctx["attr_to_label"], attr_to_cat=ctx["attr_to_cat"],
            label_to_category_map=ctx["label_to_category_map"],
            allowed_hazard_lookup=ctx["allowed_hazard_lookup"], allowed_hazard_category_lookup=ctx["allowed_hazard_category_lookup"],
            allowed_hazard_labels=ctx["allowed_hazard_labels"], allowed_hazard_category_labels=ctx["allowed_hazard_category_labels"],
            file_label_map=ctx.get("file_label_map"),
        )
        pred_labels.extend(labels)
        if _src == "mapping_file":
            file_sourced.update(labels)
    # 映射文件命中的标签保持原样，其余照旧规范化
    rest = canonicalize_pred_labels([x for x in pred_labels if x not in file_sourced],
                                    ctx["allowed_hazard_lookup"], ctx["allowed_hazard_labels"])
    return sorted(set(rest) | file_sourced)


# =========================
# 候选标签集（= 映射 keys） + label->category 推导
# =========================
def build_label_space() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """返回 (候选 label 列表, 小写->规范 label 查表, 规范 label->category 映射)。"""
    raw = load_json(LABEL_TO_CATEGORY_PATH)
    label_to_cat = {
        k.strip(): (v.strip() if isinstance(v, str) and v.strip() else k.strip())
        for k, v in raw.items()
        if isinstance(k, str) and k.strip()
    }
    label_list = sorted(label_to_cat.keys())
    label_lookup = {x.lower(): x for x in label_list}
    return label_list, label_lookup, label_to_cat


# =========================
# 统一规范化到 937-key 词表（两段共用）
# 别名表 / 过敏原食物表外置到 hazard_canon_config.json，改表只需改文件。
# =========================
def _nk_fix(x: str) -> str:
    nk = norm_key(x)
    nk = re.sub(r"\be\s+(\d{3,4})", r"e\1", nk)          # "e 102" -> "e102"
    nk = nk.replace("(", " ").replace(")", " ").replace("/", " ")
    nk = re.sub(r"\s+", " ", nk).strip()
    return nk


_CANON = {}


def build_canon(label_list: List[str]):
    _CANON["keys"] = label_list
    _CANON["lookup"] = {_nk_fix(k): k for k in label_list}
    # 权威映射文件产出的标签写法优先：936 词表里存在大小写重复
    # （Salmonella Enteritidis / Salmonella enteritidis），字典构建顺序会把小写写法覆盖上去。
    try:
        for k, v in load_json(HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH).items():
            lab = v.strip() if isinstance(v, str) and v.strip() else (k.strip() if isinstance(k, str) else "")
            if lab and _nk_fix(lab) in _CANON["lookup"]:
                _CANON["lookup"][_nk_fix(lab)] = lab
    except Exception:
        pass
    _CANON["keyset"] = set(_CANON["lookup"].keys())
    _CANON["aliases"] = {_nk_fix(k): v for k, v in _ALIASES_RAW.items()}
    _CANON["allergen"] = {_nk_fix(x) for x in _ALLERGEN_FOODS_RAW}


def canon937(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""
    nk = _nk_fix(x)
    if nk in _CANON["allergen"]:
        return "Allergens"
    if nk in _CANON["aliases"]:
        return _CANON["aliases"][nk]
    if nk in _CANON["keyset"]:
        return _CANON["lookup"][nk]
    m = re.search(r"\be(\d{3,4})", nk)                   # E 编号唯一前缀匹配
    if m:
        base = "e" + m.group(1)
        cands = {v for k, v in _CANON["lookup"].items() if k.startswith(base)}
        if len(cands) == 1:
            return next(iter(cands))
    c = canonicalize_one_label(x, _CANON["lookup"], _CANON["keys"])  # 通用兜底
    cnk = _nk_fix(c)
    if cnk in _CANON["aliases"]:
        return _CANON["aliases"][cnk]
    if cnk in _CANON["keyset"]:
        return _CANON["lookup"][cnk]
    return x


def canon_list(labels: List[str]) -> List[str]:
    out = []
    for x in labels:
        c = canon937(x)
        if c:
            out.append(c)
    return sorted(set(out))


def canon_filter(values, lookup: Dict[str, str]) -> List[str]:
    """把 LLM 输出按词表规范化（忽略大小写），丢弃不在候选集里的。"""
    if not isinstance(values, list):
        return []
    out = []
    for x in values:
        if isinstance(x, str) and x.strip():
            k = x.strip().lower()
            if k in lookup:
                out.append(lookup[k])
    return sorted(set(out))


def post_process_labels(labels: List[str]) -> List[str]:
    """移植 hazard_eval 里调好的、与 subject 无关的标签清洗规则。"""
    labels = list(labels)
    for generic, prefix in [
        ("Salmonella spp", "Salmonella "),
        ("Listeria spp", "Listeria "),
        ("Vibrio spp", "Vibrio "),
        ("Cronobacter spp", "Cronobacter "),
    ]:
        labels = remove_generic_when_specific_exists(labels, generic, prefix)
    if "Allergens" in labels and "Labelling" not in labels:
        labels = ["Allergens"]
    if any(re.match(r"^E[0-9]{3,4}", x) for x in labels) and "Food additives" in labels:
        labels = [x for x in labels if x != "Food additives"]
    if labels == ["Novel food ingredient"]:
        labels = ["Novel food"]
    if "Cannabinoid" in labels and "Novel food ingredient" in labels:
        labels = [x for x in labels if x != "Novel food ingredient"]
    return sorted(set(labels))


def derive_categories(labels: List[str], label_to_cat: Dict[str, str]) -> List[str]:
    """hazard_category_label 一律由 hazard_label 经映射推导。"""
    cats = []
    for l in labels:
        c = label_to_cat.get(l)
        if c:
            cats.append(c)
    return sorted(set(cats))


# =========================
# no_hazards 段：LLM 预测
# =========================
def predict_no_hazard_record(item, chain, prompt_variables, allowed_labels_str, label_lookup, label_to_cat) -> Tuple[List[str], List[str]]:
    product_type_value = safe_get(item, "product_type") or safe_get(item, "notification_type")
    full_payload = {
        "allowed_hazard_labels": allowed_labels_str,
        "product_type": product_type_value,
        "notification_type": product_type_value,
        "product_category": safe_get(item, "product_category"),
        "product": safe_get(item, "product"),
        "subject": safe_get(item, "subject"),
    }
    payload = {k: full_payload[k] for k in prompt_variables if k in full_payload}

    response = chain.invoke(payload)
    parsed = parse_json_output(str(response.content))
    if isinstance(parsed, list):
        parsed = {"hazard_label": parsed}

    # LLM 只预测 hazard_label；清洗 -> 统一规范化到 937 词表；category 由映射推导
    labels = canon_filter(parsed.get("hazard_label", []), label_lookup)
    labels = post_process_labels(labels)
    labels = canon_list(labels)
    cats = derive_categories(labels, label_to_cat)
    return labels, cats


# =========================
# 处理单个文件
# =========================
def predict_one_file(input_fp, ctx, chain, prompt_variables,
                     allowed_labels_str, label_lookup, label_to_cat, output_dir) -> Dict[str, int]:
    data = load_json(input_fp)
    single_record = False
    if isinstance(data, dict):
        data = [data]
        single_record = True
    if DEBUG_N is not None:
        data = data[:DEBUG_N]

    results = []
    errors = []
    n_has = n_no = 0

    for item in data:
        out_item = dict(item)
        haz = item.get("hazards", [])
        has_hazards = isinstance(haz, list) and len(haz) > 0
        try:
            if has_hazards:
                labels = canon_list(rule_predict_has_hazards(item, ctx))   # table-lookup -> 统一到 937
                cats = derive_categories(labels, label_to_cat)             # category 由映射推导
                n_has += 1
            else:
                labels, cats = predict_no_hazard_record(
                    item, chain, prompt_variables, allowed_labels_str, label_lookup, label_to_cat,
                )                                                          # LLM
                n_no += 1

            out_item["hazard_label"] = labels
            out_item["hazard_category_label"] = cats
            results.append(out_item)

        except Exception as e:
            results.append(out_item)
            errors.append({
                "file": os.path.basename(input_fp),
                "reference": item.get("reference", ""),
                "subject": item.get("subject", ""),
                "error": str(e),
            })

    pred_output_fp = input_fp if OVERWRITE_IN_PLACE else os.path.join(output_dir, os.path.basename(input_fp))
    err_output_fp = os.path.join(
        output_dir, os.path.splitext(os.path.basename(input_fp))[0] + "_hazard_prediction_errors.json"
    )
    payload_to_write = results[0] if (single_record and results) else results
    with open(pred_output_fp, "w", encoding="utf-8") as f:
        json.dump(payload_to_write, f, indent=2, ensure_ascii=False)
    if errors:
        with open(err_output_fp, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    elif os.path.exists(err_output_fp):
        os.remove(err_output_fp)

    return {"records": len(data), "errors": len(errors), "has_hazards": n_has, "no_hazards_llm": n_no}


def collect_input_files(input_dir):
    out = []
    for fp in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        name = os.path.basename(fp)
        if name in EXCLUDE_NAMES or any(name.endswith(s) for s in EXCLUDE_SUFFIXES):
            continue
        out.append(fp)
    return out


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        raise ValueError("Please set a valid AWS_BEARER_TOKEN_BEDROCK.")

    print("Building table-lookup engine (for has_hazards) ...")
    ctx = build_context()

    print("Building no_hazards LLM label space + chain ...")
    label_list, label_lookup, label_to_cat = build_label_space()
    build_canon(label_list)   # 统一规范化资源（937 词表 + alias + 过敏原折叠）
    allowed_labels_str = json.dumps(label_list, ensure_ascii=False)
    chain, prompt_variables = build_chain(NO_HAZARD_PROMPT_PATH)
    print(f"  candidate hazard_label: {len(label_list)} (= mapping keys)")
    print(f"  distinct derived categories: {len(set(label_to_cat.values()))}")
    print(f"  prompt variables: {prompt_variables}")
    print(f"  model: {CLAUDE_MODEL}")

    grand = []
    for input_dir in INPUT_DIRS:
        if not os.path.isdir(input_dir):
            print(f"[Warn] missing: {input_dir}")
            continue
        output_dir = input_dir if OVERWRITE_IN_PLACE else output_dir_for(input_dir)
        if not OVERWRITE_IN_PLACE:
            os.makedirs(output_dir, exist_ok=True)

        files = collect_input_files(input_dir)
        total_found = len(files)
        if SKIP_EXISTING and not OVERWRITE_IN_PLACE:
            files = [fp for fp in files if not os.path.exists(os.path.join(output_dir, os.path.basename(fp)))]
            print(f"[SKIP_EXISTING] {total_found - len(files)} already done, {len(files)} remaining")
        if not files:
            print(f"[Info] nothing to do in {input_dir}")
            continue

        print(f"\n=== {os.path.basename(input_dir)} -> {os.path.basename(output_dir)} ===")
        print(f"Processing {len(files)} file(s) with {MAX_WORKERS} workers")
        stats = {"has_hazards": 0, "no_hazards_llm": 0, "errors": 0}

        def _run(fp):
            return predict_one_file(
                fp, ctx, chain, prompt_variables,
                allowed_labels_str, label_lookup, label_to_cat, output_dir,
            )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_run, fp) for fp in files]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=os.path.basename(input_dir)):
                s = fut.result()
                stats["has_hazards"] += s["has_hazards"]
                stats["no_hazards_llm"] += s["no_hazards_llm"]
                stats["errors"] += s["errors"]

        dir_summary = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "files_processed": len(files),
            "has_hazards_rule": stats["has_hazards"],
            "no_hazards_llm": stats["no_hazards_llm"],
            "errors": stats["errors"],
            "model": CLAUDE_MODEL,
        }
        with open(os.path.join(output_dir, "hazard_prediction_summary_folder.json"), "w", encoding="utf-8") as f:
            json.dump(dir_summary, f, indent=2, ensure_ascii=False)
        print(f"Done: {dir_summary}")
        grand.append(dir_summary)

    print("\n===== Hazard LLM (folder) Overall =====")
    print(json.dumps(grand, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
