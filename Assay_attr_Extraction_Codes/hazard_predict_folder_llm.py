"""
混合 hazard 抽取（folder 版，单文件自包含，不 import 其它 .py）：
  - has_hazards 记录（有结构化 hazards 字段）-> 纯查表，**不调 LLM**
  - no_hazards  记录（只有 subject）        -> LLM

规格（Ron / Dr Gong）：
  * has_hazards：hazard 串取属性段（"Fragments glass - foreign bodies" -> "Fragments glass"），
    查 has_hazards_mapping_hazard_label.json 得 hazard_label（value 非空=改写，为空=key 本身）。
  * no_hazards：LLM 从候选集里指派 hazard_label，候选集 = mapping_hazard_label_to_
    hazard_category_label.json 的 keys（当前 1021 个）。
  * hazard_category_label：两段一律由该映射从 hazard_label 推导，绝不用 LLM。

预测写进已有的 hazard_label / hazard_category_label 字段，保留 product_label，
输出到新的 *_hazard_llm 文件夹。summary 里的 diagnostics 会暴露查表未命中、
以及缺 category 映射的标签。

外部依赖（数据/Prompt，非 .py）：
  - hazard/has_hazards_mapping_hazard_label.json
  - hazard/mapping_hazard_label_to_hazard_category_label.json
  - hazard_no_hazards_prompt_V3.txt（可用 PROMPT_PATH 覆盖）
  - hazard_canon_config.json
Python 包：langchain_aws, langchain_anthropic, langchain_core, tqdm
环境变量：AWS_BEARER_TOKEN_BEDROCK（Bedrock）或 ANTHROPIC_API_KEY（直连）；
         可选 AWS_REGION / CLAUDE_MODEL / RASFF_ROOT / INPUT_DIR / PROMPT_PATH
"""
import os
import re
import json
import glob
import unicodedata
from string import Formatter
import threading
from collections import Counter
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from langchain_aws import ChatBedrockConverse
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate


# =========================
# 路径 & 数据文件
# =========================
# 项目根目录：默认按本文件位置推导（仓库/容器里都对）；Docker 里用 RASFF_ROOT=/app 覆盖。
RASFF_ROOT = os.environ.get("RASFF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(RASFF_ROOT, "Assay_attr_Extraction_Codes")
HAZARD_DIR = os.path.join(RASFF_ROOT, "hazard")

HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH = os.path.join(HAZARD_DIR, "has_hazards_mapping_hazard_label.json")
MAPPING_LABEL_TO_CATEGORY_PATH = os.path.join(HAZARD_DIR, "mapping_hazard_label_to_hazard_category_label.json")

# 候选 hazard_label 集 = label->category 映射文件的 keys（937 个规范标签）
LABEL_TO_CATEGORY_PATH = MAPPING_LABEL_TO_CATEGORY_PATH

# no_hazards 段用的 prompt（外部文件）。V2 = 细粒度子标签版，对齐当前 1021 词表；
# V1 对应旧词表，在 2024 GT 上只有 0.61（V2 为 0.81），保留仅为复现历史结果。
NO_HAZARD_PROMPT_PATH = os.environ.get(
    "PROMPT_PATH", os.path.join(CODE_DIR, "hazard_no_hazards_prompt_V3.txt"))

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
# 凭证只从环境变量读（AWS_BEARER_TOKEN_BEDROCK 或 ANTHROPIC_API_KEY），不再硬编码在源码里。
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "global.anthropic.claude-sonnet-4-6")
TEMPERATURE = 0
MAX_TOKENS = 1024

# 后端选择：设置了 ANTHROPIC_API_KEY 就走 Anthropic 直连（claude-sonnet-4-6），
# 否则回退到 Bedrock（保持历史跑法可复现）。
USE_ANTHROPIC_DIRECT = bool(os.environ.get("ANTHROPIC_API_KEY"))
ANTHROPIC_MODEL = "claude-sonnet-4-6"


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
    if USE_ANTHROPIC_DIRECT:
        llm = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=120,
            max_retries=5,
        )
    else:
        llm = ChatBedrockConverse(
            model=CLAUDE_MODEL,
            region_name=AWS_REGION,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    return prompt | llm, prompt_variables


# =========================
# 规范化配置（全部外置到 hazard_canon_config.json，改规则只需改文件）
#   aliases             -> canon937 别名表（变体 -> 937 标准标签）
#   allergen_foods      -> 过敏原食物名（出现即折叠为 Allergens）
#   rule_manual_aliases -> 规则引擎 canonicalize_one_label 的别名表
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
# 文本规范化（两段共用）
# =========================
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


def remove_generic_when_specific_exists(labels: List[str], generic_label: str, prefix: str) -> List[str]:
    has_specific = any(x.startswith(prefix) and x != generic_label for x in labels)
    if has_specific and generic_label in labels:
        labels = [x for x in labels if x != generic_label]
    return labels


class _SafeCounter:
    """诊断计数器。predict_one_file 在线程池里并发跑，共享同一个 ctx，故加锁。"""

    def __init__(self):
        self._c = Counter()
        self._lock = threading.Lock()

    def __setitem__(self, key, value):
        with self._lock:
            self._c[key] = value

    def __getitem__(self, key):
        with self._lock:
            return self._c[key]

    def as_dict(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._c)


def build_has_hazard_label_map(inventory: Dict[str, Any]) -> Dict[str, str]:
    """权威映射表 has_hazards_mapping_hazard_label.json -> {规范化键: hazard_label}。

    文件里两种写法：value 非空 = 改写（Fragments glass -> Foreign bodies (glass)）；
    value 为空 = 恒等（标签即 key 本身，如 Cyanide / Salmonella Altona）。
    """
    out: Dict[str, str] = {}
    for k, v in inventory.items():
        if not isinstance(k, str) or not k.strip():
            continue
        label = v.strip() if isinstance(v, str) and v.strip() else k.strip()
        out.setdefault(norm_key(k), label)
    return out


def build_context() -> Dict[str, Any]:
    """构建 has_hazards 段所需的查表（一次性）。

    规格（Ron / Dr Gong）：有 hazards 数据的记录**不用 LLM**，hazard_label 由
    has_hazards_mapping_hazard_label.json 查表得出，hazard_category_label 再由
    mapping_hazard_label_to_hazard_category_label.json 从 hazard_label 推导。

    历史上这里还挂了一层从 rasff_data_2020_to_2026_*_with_labels.json 统计出来的
    兜底表（raw_hazard_to_label / attr_to_label 等）。那两份 gold 语料已被删除，
    这层兜底既无法重建、也超出上述「只查表」规格，故移除；仅保留只依赖 1021 词表的
    canon937 兜底作为安全网（见 rule_predict_has_hazards）。在 2020–2026 全量
    has_hazards 语料的 32350 条 hazard 上，映射文件命中率为 100.000%，安全网不触发。
    """
    return {
        "file_label_map": build_has_hazard_label_map(load_json(HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH)),
        "stats": _SafeCounter(),
    }


def rule_predict_has_hazards(item: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
    """has_hazards 段：逐条 hazards -> hazard_label（纯查表，不调 LLM）。

    ① hazard 串先经 extract_hazard_attribute 取属性段（"Fragments glass - foreign
       bodies" -> "Fragments glass"），查映射文件；未命中再用 hazard 整串查。
    ② 安全网：映射文件没有这条写法时，用 canon937 往 1021 词表靠；靠不上就跳过
       （宁可留空也不猜，避免污染输出）。触发次数记进 ctx["stats"]，会写入 summary。
    """
    stats = ctx.get("stats")
    file_label_map = ctx["file_label_map"]
    labels = []
    for h in item.get("hazards", []):
        if not isinstance(h, dict):
            continue
        raw = h.get("hazard", "") or ""
        if not str(raw).strip():
            continue
        attr = extract_hazard_attribute(raw)
        lab = file_label_map.get(norm_key(attr)) or file_label_map.get(norm_key(raw))
        if lab:
            labels.append(lab)
            if stats is not None:
                stats["mapping_file"] += 1
            continue
        fallback = canon_list([attr])                      # 只依赖词表，不依赖已删除的 gold
        fallback = [x for x in fallback if _nk_fix(x) in (_CANON.get("keyset") or set())]
        if fallback:
            labels.extend(fallback)
            if stats is not None:
                stats["canon_fallback"] += 1
        elif stats is not None:
            stats["unmapped_hazard"] += 1
    return sorted(set(labels))


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
# 别名表 / 过敏原食物表外置到 hazard_canon_config.json，改规则只需改文件。
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
    # 别名表是按旧的 936 词表写的，部分目标写法在 1021 词表里已经不存在
    # （"moulds" -> "Mould"，而现在词表只有 "Moulds"）。canon937 命中别名后是无条件
    # 返回的，于是这些别名会把已经正确的标签改错。这里先按词表校准一次：目标不在
    # 词表内的整条丢掉，让原标签自己走后面的匹配逻辑。不改 hazard_canon_config.json。
    aliases = {}
    for k, v in _ALIASES_RAW.items():
        nk = _nk_fix(k)
        if nk and isinstance(v, str) and v.strip() and _nk_fix(v) in _CANON["keyset"]:
            aliases[nk] = _CANON["lookup"][_nk_fix(v)]
    _CANON["aliases"] = aliases
    _CANON["allergen"] = {_nk_fix(x) for x in _ALLERGEN_FOODS_RAW}


def canon937(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""
    nk = _nk_fix(x)
    if nk in _CANON["allergen"]:
        return "Allergens"
    # 词表命中优先于别名表：别名表里有 "nicotinamide mononucleotide (nmn)" -> "Novel food"
    # 这种把已经合法的具体标签折叠回旧兜底标签的条目。已在 1021 词表里的一律不改写。
    if nk in _CANON["keyset"]:
        return _CANON["lookup"][nk]
    if nk in _CANON["aliases"]:
        return _CANON["aliases"][nk]
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
    """规范化一组标签。别名表（hazard_canon_config.json）里存在指向**已废弃标签**的死链
    （例如 Moulds -> Mould，而 Mould 已不在词表里），套用后会被下游按“词表外”丢掉，
    等于把对的答案改成错的。因此只在规范化结果仍落在词表内时才采用，否则保留原标签。"""
    keyset = _CANON.get("keyset") or set()
    out = []
    for x in labels:
        if not isinstance(x, str) or not x.strip():
            continue
        c = canon937(x)
        if not c:
            continue
        out.append(c if (not keyset or _nk_fix(c) in keyset) else x.strip())
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


def derive_categories(labels: List[str], label_to_cat: Dict[str, str], stats=None) -> List[str]:
    """hazard_category_label 一律由 hazard_label 经映射推导（绝不用 LLM）。

    映射文件里查不到 category 的标签会让 hazard_category_label 变空——这是数据问题
    （如 Bacterial contamination / Not in catalogue / Toxin unknown 不在映射 keys 里），
    不是代码问题，所以记进 stats 让它在 summary 里可见，而不是静默丢掉。
    """
    nk_to_cat = {norm_key(k): v for k, v in label_to_cat.items()}
    cats = []
    for l in labels:
        c = label_to_cat.get(l) or nk_to_cat.get(norm_key(l))
        if c:
            cats.append(c)
        elif stats is not None:
            stats["label_without_category:" + l] += 1
    return sorted(set(cats))


# =========================
# no_hazards 段：LLM 预测
# =========================
def predict_no_hazard_record(item, chain, prompt_variables, allowed_labels_str, label_lookup, label_to_cat,
                             stats=None) -> Tuple[List[str], List[str]]:
    product_type_value = (
        safe_get(item, "product_type")
        or safe_get(item, "notification_product_type")   # ASEAN: 真实产品类型（Food/Feed/...）
        or safe_get(item, "notification_type")
    )
    # 兼容 ASEAN ARASFF：优先使用显式 hazard 字符串字段作为主信号，subject 作为补充。
    # （RASFF 无 hazard 字段时自动退回只用 subject，行为不变。）
    raw_hazard = safe_get(item, "hazard").strip()
    base_subject = safe_get(item, "subject")
    if raw_hazard:
        subject_value = f"Reported hazard(s): {raw_hazard}\n\nNotification subject: {base_subject}"
    else:
        subject_value = base_subject
    full_payload = {
        "allowed_hazard_labels": allowed_labels_str,
        "product_type": product_type_value,
        "notification_type": product_type_value,
        "product_category": safe_get(item, "product_category"),
        "product": safe_get(item, "product"),
        "subject": subject_value,
    }
    payload = {k: full_payload[k] for k in prompt_variables if k in full_payload}

    response = chain.invoke(payload)
    parsed = parse_json_output(str(response.content))
    if isinstance(parsed, list):
        parsed = {"hazard_label": parsed}

    # LLM 只预测 hazard_label。顺序与已验证的评估脚本一致：
    # 先 canon_list 把旧词表写法救回来，再过词表丢掉词表外的，最后清洗。
    # （过词表之后不能再 canon_list：此时标签已是词表成员，再规范化只会把它踢出词表。）
    labels = canon_list(parsed.get("hazard_label", []) or [])
    labels = canon_filter(labels, label_lookup)
    labels = post_process_labels(labels)
    cats = derive_categories(labels, label_to_cat, stats)
    return labels, cats


# =========================
# 处理单个文件
# =========================
def predict_one_file(input_fp, ctx, chain, prompt_variables,
                     allowed_labels_str, label_lookup, label_to_cat, output_dir) -> Dict[str, int]:
    data = load_json(input_fp)
    # 空/None 通知：原样写出、跳过标注，不中断整批
    if data is None:
        out_fp = input_fp if OVERWRITE_IN_PLACE else os.path.join(output_dir, os.path.basename(input_fp))
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(None, f, ensure_ascii=False, indent=2)
        return {"records": 0, "errors": 0, "has_hazards": 0, "no_hazards_llm": 0}
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
                # 纯查表，不调 LLM。映射文件给出的就是权威写法，不再二次规范化。
                labels = rule_predict_has_hazards(item, ctx)
                cats = derive_categories(labels, label_to_cat, ctx.get("stats"))
                n_has += 1
            else:
                labels, cats = predict_no_hazard_record(
                    item, chain, prompt_variables, allowed_labels_str, label_lookup, label_to_cat,
                    ctx.get("stats"),
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
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")):
        raise ValueError("Set ANTHROPIC_API_KEY (Anthropic direct) or AWS_BEARER_TOKEN_BEDROCK (Bedrock).")

    print("Building label space + normalization ...")
    label_list, label_lookup, label_to_cat = build_label_space()
    build_canon(label_list)   # 统一规范化资源（词表 + alias + 过敏原折叠）

    print("Building has_hazards lookup table (no LLM) ...")
    ctx = build_context()

    print("Building no_hazards LLM chain ...")
    allowed_labels_str = json.dumps(label_list, ensure_ascii=False)
    chain, prompt_variables = build_chain(NO_HAZARD_PROMPT_PATH)
    print(f"  candidate hazard_label: {len(label_list)} (= mapping keys)")
    print(f"  distinct derived categories: {len(set(label_to_cat.values()))}")
    print(f"  has_hazards mapping keys: {len(ctx['file_label_map'])}")
    print(f"  prompt: {os.path.basename(NO_HAZARD_PROMPT_PATH)} | variables: {prompt_variables}")
    print(f"  model: {CLAUDE_MODEL} @ {AWS_REGION}"
          f"{' (Anthropic direct)' if USE_ANTHROPIC_DIRECT else ' (Bedrock)'}")

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

        diag = ctx["stats"].as_dict()
        dir_summary = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "files_processed": len(files),
            "has_hazards_rule": stats["has_hazards"],
            "no_hazards_llm": stats["no_hazards_llm"],
            "errors": stats["errors"],
            "model": CLAUDE_MODEL,
            "prompt": os.path.basename(NO_HAZARD_PROMPT_PATH),
            # 诊断：mapping_file=查表命中；canon_fallback=映射文件没这条写法、走了词表安全网；
            # unmapped_hazard=两者都不行（label 留空）；label_without_category:X=X 缺 category 映射。
            "diagnostics": diag,
        }
        for key, msg in (("canon_fallback", "has_hazards 有 hazard 写法不在映射文件里，走了词表安全网"),
                         ("unmapped_hazard", "has_hazards 有 hazard 完全无法映射，hazard_label 留空")):
            if diag.get(key):
                print(f"[Warn] {msg}: {diag[key]} 条 —— 建议补进 has_hazards_mapping_hazard_label.json")
        nocat = {k.split(":", 1)[1]: v for k, v in diag.items() if k.startswith("label_without_category:")}
        if nocat:
            print(f"[Warn] 这些 hazard_label 在 mapping_hazard_label_to_hazard_category_label.json 里"
                  f"没有 category，对应记录的 hazard_category_label 会是空: {nocat}")
        with open(os.path.join(output_dir, "hazard_prediction_summary_folder.json"), "w", encoding="utf-8") as f:
            json.dump(dir_summary, f, indent=2, ensure_ascii=False)
        print(f"Done: {dir_summary}")
        grand.append(dir_summary)

    print("\n===== Hazard LLM (folder) Overall =====")
    print(json.dumps(grand, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
