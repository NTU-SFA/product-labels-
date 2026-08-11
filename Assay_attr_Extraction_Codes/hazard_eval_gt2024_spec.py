"""
按最新规格在 2024 ground-truth 上评估 hazard_label / hazard_category_label 抽取准确率。

规格（用户给定）：
  * has_hazards 记录：hazard 字段 -> has_hazards_mapping_hazard_label.json 查表得 hazard_label
    （value 非空=改写；value 为空=标签即 key 本身）。不用 LLM。
  * no_hazards 记录：用 LLM 从候选集里指派 hazard_label，
    候选集 = mapping_hazard_label_to_hazard_category_label.json 的 keys（当前 1021 个）。
  * hazard_category_label：两段一律由 mapping_hazard_label_to_hazard_category_label.json
    从 hazard_label 推导，绝不用 LLM。

与 hazard_eval_gt2024.py 的区别：不依赖已被删除的
hazard/rasff_data_2020_to_2026_*_with_labels.json 语料统计表（那条路径既跑不起来，
也会因为 gold 里含 2024 而泄漏），has_hazards 段只走映射文件。

LLM 缓存存的是**原始**输出（未过词表），换词表后可直接复用：
  Outputs_hazard_eval_gt2024_spec/llm_raw_cache.json
Bedrock prompt caching 默认开启（词表前缀 ~6.5k tokens，可省约 90% 输入费用）。
"""
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import hazard_predict_folder_llm as L

RASFF_ROOT = L.RASFF_ROOT
GT_PATH = os.environ.get("GROUND_TRUTH_PATH", os.path.join(RASFF_ROOT, "rasff_2024_ground_truth_labels.json"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(L.CODE_DIR, "Outputs_hazard_eval_gt2024_spec"))
CACHE_PATH = os.path.join(OUTPUT_DIR, "llm_raw_cache.json")
# 换 prompt 请配 PROMPT_PATH（V2 = 细粒度子标签版，对齐 1021 词表）
PROMPT_PATH = os.environ.get("PROMPT_PATH", L.NO_HAZARD_PROMPT_PATH)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
USE_CACHE = os.environ.get("USE_CACHE", "1") == "1"
USE_PROMPT_CACHE = os.environ.get("USE_PROMPT_CACHE", "1") == "1"
APPLY_POST_PROCESS = os.environ.get("APPLY_POST_PROCESS", "1") == "1"
PRE_CANON = os.environ.get("PRE_CANON", "0") == "1"


# ---------- has_hazards：纯映射 ----------
def build_has_hazards_label_map():
    inv = L.load_json(L.HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH)
    out = {}
    for k, v in inv.items():
        if not isinstance(k, str) or not k.strip():
            continue
        out.setdefault(L.norm_key(k), (v.strip() if isinstance(v, str) and v.strip() else k.strip()))
    return out


def predict_has_hazards(rec, file_label_map):
    labels = []
    for h in rec.get("hazards", []):
        if not isinstance(h, dict):
            continue
        raw = h.get("hazard", "") or ""
        attr = L.extract_hazard_attribute(raw)
        lab = file_label_map.get(L.norm_key(attr)) or file_label_map.get(L.norm_key(raw))
        if lab:
            labels.append(lab)
    return sorted(set(labels))


def derive_categories(labels, label_to_cat, nk_to_cat, unmapped=None):
    cats = []
    for l in labels:
        c = label_to_cat.get(l) or nk_to_cat.get(L.norm_key(l))
        if c:
            cats.append(c)
        elif unmapped is not None:
            unmapped.append(l)
    return sorted(set(cats))


# ---------- no_hazards：LLM ----------
def build_cached_invoker(prompt_text, prompt_variables, label_list):
    """把 prompt 渲染后按词表末尾切成两块，中间插 Bedrock cachePoint。"""
    from langchain_aws import ChatBedrockConverse

    llm = ChatBedrockConverse(
        model=L.CLAUDE_MODEL, region_name=L.AWS_REGION,
        temperature=L.TEMPERATURE, max_tokens=L.MAX_TOKENS,
    )
    labels_str = json.dumps(label_list, ensure_ascii=False)

    def invoke(payload):
        text = prompt_text.format(**{k: payload.get(k, "") for k in prompt_variables})
        idx = text.find(labels_str)
        if idx < 0:
            return llm.invoke(text).content
        split = idx + len(labels_str)
        blocks = [{"text": text[:split]}, {"cachePoint": {"type": "default"}}, {"text": text[split:]}]
        return llm.invoke([{"role": "user", "content": blocks}]).content

    return invoke


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label_list, label_lookup, label_to_cat = L.build_label_space()
    L.build_canon(label_list)
    nk_to_cat = {L.norm_key(k): v for k, v in label_to_cat.items()}
    file_label_map = build_has_hazards_label_map()
    print(f"label space = {len(label_list)} | has_hazards mapping keys = {len(file_label_map)}")

    gt = L.load_json(GT_PATH)
    has_recs = [r for r in gt if isinstance(r.get("hazards"), list) and r.get("hazards")]
    no_recs = [r for r in gt if not (isinstance(r.get("hazards"), list) and r.get("hazards"))]
    print(f"GT total={len(gt)} | has_hazards={len(has_recs)} | no_hazards={len(no_recs)}")

    # ---- LLM 预测（只跑 no_hazards，存原始输出）----
    cache = L.load_json(CACHE_PATH) if (USE_CACHE and os.path.exists(CACHE_PATH)) else {}
    print(f"cached raw LLM preds: {len(cache)}")

    prompt_text = L.load_text(PROMPT_PATH)
    prompt_variables = L.extract_template_variables(prompt_text)
    labels_str = json.dumps(label_list, ensure_ascii=False)
    if USE_PROMPT_CACHE and not L.USE_ANTHROPIC_DIRECT:
        invoke = build_cached_invoker(prompt_text, prompt_variables, label_list)
    else:
        chain, _ = L.build_chain(PROMPT_PATH)
        invoke = lambda payload: chain.invoke({k: payload.get(k, "") for k in prompt_variables}).content

    def run_llm_raw(r):
        ptype = L.safe_get(r, "product_type") or L.safe_get(r, "notification_type")
        payload = {
            "allowed_hazard_labels": labels_str, "product_type": ptype, "notification_type": ptype,
            "product_category": L.safe_get(r, "product_category"), "product": L.safe_get(r, "product"),
            "subject": L.safe_get(r, "subject"),
        }
        try:
            parsed = L.parse_json_output(str(invoke(payload)))
            if isinstance(parsed, list):
                parsed = {"hazard_label": parsed}
            out = parsed.get("hazard_label", [])
            return [x for x in out if isinstance(x, str)] if isinstance(out, list) else []
        except Exception as e:
            return {"__error__": f"{type(e).__name__}: {e}"[:200]}

    todo = [r for r in no_recs if str(r.get("reference", "")) not in cache]
    print(f"LLM calls needed: {len(todo)} (workers={MAX_WORKERS}, prompt_cache={USE_PROMPT_CACHE})")
    if todo:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(run_llm_raw, r): str(r.get("reference", "")) for r in todo}
            done = 0
            for fut in as_completed(futs):
                cache[futs[fut]] = fut.result()
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
                    json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False)
        json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False)
    n_err = sum(1 for v in cache.values() if isinstance(v, dict))
    print(f"LLM errors in cache: {n_err}")

    # ---- 打分 ----
    M = {(f, s): Metric() for f in ("label", "cat") for s in ("has", "no", "all")}
    mismatches, unmapped_labels = [], []

    for r in gt:
        ref = str(r.get("reference", ""))
        is_has = isinstance(r.get("hazards"), list) and bool(r.get("hazards"))
        if is_has:
            labels = predict_has_hazards(r, file_label_map)
        else:
            raw = cache.get(ref, [])
            raw = [] if isinstance(raw, dict) else raw
            if PRE_CANON:            # 先 canon937 再过词表：可把旧词表写法救回新词表
                raw = L.canon_list(raw)
            labels = L.canon_filter(raw, label_lookup)
            # 过词表之后不再 canon_list：此时标签已是词表成员，再规范化只会把它踢出词表
            if APPLY_POST_PROCESS:
                labels = L.post_process_labels(labels)
        cats = derive_categories(labels, label_to_cat, nk_to_cat, unmapped_labels)

        pred_l, truth_l = norm_set(labels), norm_set(r.get("hazard_label") or [])
        pred_c, truth_c = norm_set(cats), norm_set(r.get("hazard_category_label") or [])
        seg = "has" if is_has else "no"
        for f, p, t in (("label", pred_l, truth_l), ("cat", pred_c, truth_c)):
            M[(f, seg)].add(p, t)
            M[(f, "all")].add(p, t)
        if pred_l != truth_l or pred_c != truth_c:
            mismatches.append({
                "reference": ref, "segment": seg, "subject": r.get("subject", ""),
                "hazards_raw": [h.get("hazard") for h in (r.get("hazards") or []) if isinstance(h, dict)],
                "gold_label": sorted(truth_l), "pred_label": sorted(pred_l),
                "gold_category": sorted(truth_c), "pred_category": sorted(pred_c),
            })

    report = {
        "gt_path": GT_PATH, "total": len(gt),
        "config": {"label_space": len(label_list), "post_process": APPLY_POST_PROCESS,
                   "prompt": os.path.basename(PROMPT_PATH), "pre_canon": PRE_CANON,
                   "llm_model": L.CLAUDE_MODEL, "llm_errors": n_err},
        "hazard_label": {s: M[("label", s)].report() for s in ("has", "no", "all")},
        "hazard_category_label": {s: M[("cat", s)].report() for s in ("has", "no", "all")},
        "labels_missing_from_label_to_category_mapping": sorted(set(unmapped_labels)),
    }
    json.dump(report, open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(mismatches, open(os.path.join(OUTPUT_DIR, "mismatches.json"), "w"), ensure_ascii=False, indent=2)

    for field in ("hazard_label", "hazard_category_label"):
        print(f"\n==== {field} ====")
        hdr = f"{'segment':8} {'n':>6} {'exact_acc':>10} {'jaccard':>8} {'P':>7} {'R':>7} {'F1':>7}"
        print(hdr); print("-" * len(hdr))
        for s, name in (("has", "has_haz"), ("no", "no_haz"), ("all", "OVERALL")):
            m = report[field][s]
            print(f"{name:8} {m['n']:>6} {m['exact_accuracy']:>10} {m['mean_jaccard']:>8} "
                  f"{m['micro_precision']:>7} {m['micro_recall']:>7} {m['micro_f1']:>7}")
    print(f"\nmismatches={len(mismatches)} | unmapped labels={report['labels_missing_from_label_to_category_mapping']}")
    print(f"Saved: {OUTPUT_DIR}/metrics.json + mismatches.json + llm_raw_cache.json")


def norm_set(labels):
    return {x.strip().lower() for x in labels if isinstance(x, str) and x.strip()}


class Metric:
    def __init__(self):
        self.n = self.exact = 0
        self.jac = 0.0
        self.tp = self.fp = self.fn = 0

    def add(self, pred, truth):
        self.n += 1
        if pred == truth:
            self.exact += 1
        inter, union = pred & truth, pred | truth
        self.jac += (len(inter) / len(union)) if union else 1.0
        self.tp += len(inter)
        self.fp += len(pred - truth)
        self.fn += len(truth - pred)

    def report(self):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return {
            "n": self.n,
            "exact_accuracy": round(self.exact / self.n, 4) if self.n else 0.0,
            "mean_jaccard": round(self.jac / self.n, 4) if self.n else 0.0,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "micro_precision": round(p, 4), "micro_recall": round(r, 4),
            "micro_f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
        }


if __name__ == "__main__":
    main()
