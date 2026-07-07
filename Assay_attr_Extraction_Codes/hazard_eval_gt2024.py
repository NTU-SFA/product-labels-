"""
用当前混合抽取逻辑（has_hazards->规则，no_hazards->LLM，canon937，映射推导 category）
在 ground-truth 文件上评估 hazard_label 与 hazard_category_label 的准确率 / F1 等指标。

GT 文件：/Users/xueluangong/Desktop/GPT Source Codes/rasff_2024_ground_truth_labels.json
（自带 hazards/subject 作输入，hazard_label/hazard_category_label 作标注）

指标：exact-match accuracy（整条集合相等）、mean Jaccard、micro precision/recall/F1，
分 has_hazards / no_hazards / overall 三段，label 与 category 各算一套。

LLM 预测带缓存（Outputs_hazard_eval_gt2024/llm_preds_cache.json），重跑不重复调用。
"""
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import hazard_predict_folder_llm as L

RASFF_ROOT = os.environ.get("RASFF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_PATH = os.environ.get(
    "GROUND_TRUTH_PATH",
    os.path.join(RASFF_ROOT, "rasff_2024_ground_truth_labels.json"),
)
OUTPUT_DIR = os.path.join(L.CODE_DIR, "Outputs_hazard_eval_gt2024")
CACHE_PATH = os.path.join(OUTPUT_DIR, "llm_preds_cache.json")
MAX_WORKERS = 10
USE_CACHE = True


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
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {
            "n": self.n,
            "exact_accuracy": round(self.exact / self.n, 4) if self.n else 0.0,
            "mean_jaccard": round(self.jac / self.n, 4) if self.n else 0.0,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "micro_precision": round(p, 4),
            "micro_recall": round(r, 4),
            "micro_f1": round(f1, 4),
        }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building rule engine + label space + LLM chain ...")
    ctx = L.build_context()
    label_list, label_lookup, label_to_cat = L.build_label_space()
    L.build_canon(label_list)
    allowed_labels_str = json.dumps(label_list, ensure_ascii=False)
    chain, pv = L.build_chain(L.NO_HAZARD_PROMPT_PATH)

    gt = L.load_json(GT_PATH)
    has_recs = [r for r in gt if isinstance(r.get("hazards"), list) and len(r.get("hazards")) > 0]
    no_recs = [r for r in gt if not (isinstance(r.get("hazards"), list) and len(r.get("hazards")) > 0)]
    print(f"GT total={len(gt)} | has_hazards={len(has_recs)} | no_hazards={len(no_recs)}")

    # ---- no_hazards 段：LLM 预测 hazard_label（原始，带缓存）----
    cache = {}
    if USE_CACHE and os.path.exists(CACHE_PATH):
        cache = L.load_json(CACHE_PATH)
        print(f"Loaded {len(cache)} cached LLM preds")

    def run_llm_raw(r):
        product_type_value = L.safe_get(r, "product_type") or L.safe_get(r, "notification_type")
        full = {
            "allowed_hazard_labels": allowed_labels_str, "product_type": product_type_value,
            "notification_type": product_type_value, "product_category": L.safe_get(r, "product_category"),
            "product": L.safe_get(r, "product"), "subject": L.safe_get(r, "subject"),
        }
        payload = {k: full[k] for k in pv if k in full}
        try:
            parsed = L.parse_json_output(str(chain.invoke(payload).content))
            if isinstance(parsed, list):
                parsed = {"hazard_label": parsed}
            return L.canon_filter(parsed.get("hazard_label", []), label_lookup)
        except Exception:
            return []

    todo = [r for r in no_recs if str(r.get("reference", "")) not in cache]
    print(f"Need LLM for {len(todo)} no_hazards records (workers={MAX_WORKERS}) ...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_llm_raw, r): str(r.get("reference", "")) for r in todo}
        done = 0
        for fut in as_completed(futs):
            cache[futs[fut]] = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}")
                json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False)
    json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False)

    # ---- 组装预测并打分 ----
    M = {("label", s): Metric() for s in ("has", "no", "all")}
    M.update({("cat", s): Metric() for s in ("has", "no", "all")})
    mismatches = []

    for r in gt:
        ref = str(r.get("reference", ""))
        hz = r.get("hazards", [])
        is_has = isinstance(hz, list) and len(hz) > 0

        if is_has:
            labels = L.canon_list(L.rule_predict_has_hazards(r, ctx))
        else:
            labels = L.canon_list(L.post_process_labels(L.canon_filter(cache.get(ref, []), label_lookup)))
        cats = L.derive_categories(labels, label_to_cat)

        pred_l, truth_l = norm_set(labels), norm_set(r.get("hazard_label", []))
        pred_c, truth_c = norm_set(cats), norm_set(r.get("hazard_category_label", []))

        seg = "has" if is_has else "no"
        M[("label", seg)].add(pred_l, truth_l)
        M[("label", "all")].add(pred_l, truth_l)
        M[("cat", seg)].add(pred_c, truth_c)
        M[("cat", "all")].add(pred_c, truth_c)

        if pred_l != truth_l or pred_c != truth_c:
            mismatches.append({
                "reference": ref, "segment": seg, "subject": r.get("subject", ""),
                "gold_label": sorted(truth_l), "pred_label": sorted(pred_l),
                "gold_category": sorted(truth_c), "pred_category": sorted(pred_c),
            })

    report = {
        "gt_path": GT_PATH,
        "total": len(gt),
        "hazard_label": {s: M[("label", s)].report() for s in ("has", "no", "all")},
        "hazard_category_label": {s: M[("cat", s)].report() for s in ("has", "no", "all")},
    }
    json.dump(report, open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(mismatches, open(os.path.join(OUTPUT_DIR, "mismatches.json"), "w"), ensure_ascii=False, indent=2)

    # ---- 打印 ----
    for field in ("hazard_label", "hazard_category_label"):
        print(f"\n==== {field} ====")
        hdr = f"{'segment':8} {'n':>6} {'exact_acc':>10} {'jaccard':>8} {'P':>7} {'R':>7} {'F1':>7}"
        print(hdr); print("-" * len(hdr))
        for s, name in [("has", "has_haz"), ("no", "no_haz"), ("all", "OVERALL")]:
            m = report[field][s]
            print(f"{name:8} {m['n']:>6} {m['exact_accuracy']:>10} {m['mean_jaccard']:>8} "
                  f"{m['micro_precision']:>7} {m['micro_recall']:>7} {m['micro_f1']:>7}")
    print(f"\nSaved: {OUTPUT_DIR}/metrics.json + mismatches.json + llm_preds_cache.json")


if __name__ == "__main__":
    main()
