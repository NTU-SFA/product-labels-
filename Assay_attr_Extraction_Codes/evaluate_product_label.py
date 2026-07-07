# -*- coding: utf-8 -*-
"""
用 Claude_predict_folder.py 里的规则（同一个 prompt / 同一套候选标签）对
rasff_2024_ground_truth_labels.json 做 product_label 抽取，然后和文件里自带的
ground-truth product_label 对比，计算 Precision / Recall / F1。

特点：
- 直接复用 Claude_predict_folder 的 build_chain / predict_one_record，规则完全一致。
- 多线程并发预测（默认 8 线程）。
- 预测结果带缓存：中途断掉/重跑只补没跑过的记录（按 reference / 行号）。

两套打分（每个子集都给）：
- standard（标准 micro）：多预测的标签算 FP，会拉低 Precision。
    Precision=TP/(TP+FP)  Recall=TP/(TP+FN)  F1=2PR/(P+R)
- tolerant（容忍多预测）：只要 ground-truth 标签 ⊆ 预测标签，这条就算对；
    多预测不罚。subset_match_acc = #(gt ⊆ pred) / N

子集划分：
- overall              全部
- single_label         真值恰好 1 个标签
- multiple_labels      真值 >=2 个标签
- animal_feed          product_type == "Feed"

- food_contact_material product_type == "Food contact material"
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# 让脚本无论从哪里运行都能 import 到同目录的 Claude_predict_folder
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# 复用预测规则（import 时会执行其顶层配置，但不会跑 main()）
import Claude_predict_folder as P


# =========================
# 路径配置
# =========================
RASFF_ROOT = os.environ.get("RASFF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.environ.get(
    "GROUND_TRUTH_PATH",
    os.path.join(RASFF_ROOT, "rasff_2024_ground_truth_labels.json"),
)
OUTPUT_DIR = os.path.join(THIS_DIR, "eval_product_label_output")
PRED_CACHE_PATH = os.path.join(OUTPUT_DIR, "predictions.json")     # 逐条预测明细（含缓存）
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")           # 指标汇总


# =========================
# 工具
# =========================
def record_key(item, idx):
    """每条记录的唯一键：优先 reference，没有就用行号。"""
    ref = item.get("reference")
    if ref:
        return str(ref)
    return f"__idx_{idx}"


def normalize_gt(labels):
    """真值标签也用和预测一样的归一化（lower / strip / 去重 / 排序）。"""
    return P.normalize_labels(labels if isinstance(labels, list) else [])


def load_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {row["key"]: row for row in data}
        except Exception:
            return {}
    return {}


def save_cache(path, cache):
    rows = list(cache.values())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# =========================
# 打分
# =========================
def score(pairs):
    """
    Tolerant micro Precision / Recall / F1。

    规则：只要某条记录的 ground-truth 标签都被预测到了（gt ⊆ pred），
    多预测的标签就不算错（FP=0）；只有当这条记录漏了 GT 标签时，
    多出来的预测才算 FP。Recall 不受影响（漏的 GT 永远算 FN）。

      每条:  tp = |gt ∩ pred| ,  fn = |gt - pred|
             fp = 0              若 gt ⊆ pred（全部命中，多预测免罚）
             fp = |pred - gt|    否则（漏了 GT，多预测才算错）
    """
    n = len(pairs)
    tp = fp = fn = 0
    for gt, pr in pairs:
        tp += len(gt & pr)
        fn += len(gt - pr)
        if not (gt <= pr):                # 漏了 GT 才罚多预测
            fp += len(pr - gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"n": n, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# =========================
# 预测（带缓存）
# =========================
def run_predictions(data, args):
    available_labels = P.normalize_labels(P.load_labels(P.PRODUCT_LABELS_PATH))
    available_set = set(available_labels)
    used_labels_str = json.dumps(available_labels, ensure_ascii=False)
    extra_labels_str = json.dumps([], ensure_ascii=False)
    available_labels_str = json.dumps(available_labels, ensure_ascii=False)

    chain, prompt_variables = P.build_chain(P.PROMPT_PATH)

    cache = {} if args.no_cache else load_cache(PRED_CACHE_PATH)
    if cache:
        print(f"Loaded {len(cache)} cached predictions from {PRED_CACHE_PATH}")

    todo = []
    for idx, item in enumerate(data):
        key = record_key(item, idx)
        if key in cache and "pred" in cache[key]:
            continue
        todo.append((idx, item, key))

    print(f"To predict now: {len(todo)} (cached/reused: {len(data) - len(todo)})")

    def work(idx, item, key):
        pred = P.predict_one_record(
            item, chain, prompt_variables,
            used_labels_str, extra_labels_str, available_labels_str, available_set,
        )
        return idx, item, key, pred

    errors = 0
    if todo:
        done_since_save = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, idx, item, key) for idx, item, key in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Predicting"):
                try:
                    idx, item, key, pred = fut.result()
                except Exception:
                    errors += 1
                    continue
                cache[key] = {
                    "key": key,
                    "reference": item.get("reference", ""),
                    "product_type": item.get("product_type", ""),
                    "product": item.get("product", ""),
                    "subject": item.get("subject", ""),
                    "gt": normalize_gt(item.get("product_label")),
                    "pred": pred,
                }
                done_since_save += 1
                if done_since_save >= 50:
                    save_cache(PRED_CACHE_PATH, cache)
                    done_since_save = 0
        save_cache(PRED_CACHE_PATH, cache)

    if errors:
        print(f"[Warn] {errors} records failed to predict (skipped in metrics).")
    return cache


# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只评测前 N 条")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数，默认 8")
    ap.add_argument("--no-cache", action="store_true", help="忽略缓存，全部重新预测")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        raise ValueError("Please set a valid AWS_BEARER_TOKEN_BEDROCK.")

    data = P.load_json(GROUND_TRUTH_PATH)
    if not isinstance(data, list):
        data = [data]
    if args.limit is not None:
        data = data[: args.limit]

    print(f"Ground truth: {len(data)} records | Model: {P.CLAUDE_MODEL}")

    cache = run_predictions(data, args)

    # ---- 组装 (gt, pred) 并打子集标签 ----
    SUBSETS = ["overall", "single_label", "multiple_labels",
               "animal_feed", "food_contact_material"]
    buckets = {s: [] for s in SUBSETS}

    evaluated = 0
    for idx, item in enumerate(data):
        row = cache.get(record_key(item, idx))
        if not row or "pred" not in row:
            continue
        evaluated += 1
        gt = set(normalize_gt(item.get("product_label")))
        pr = set(row["pred"])
        ptype = (item.get("product_type") or "").strip().lower()

        buckets["overall"].append((gt, pr))
        if len(gt) == 1:
            buckets["single_label"].append((gt, pr))
        elif len(gt) >= 2:
            buckets["multiple_labels"].append((gt, pr))
        if ptype == "feed":
            buckets["animal_feed"].append((gt, pr))
        elif ptype == "food contact material":
            buckets["food_contact_material"].append((gt, pr))

    results = {s: score(buckets[s]) for s in SUBSETS}

    metrics = {
        "model": P.CLAUDE_MODEL,
        "prompt_path": P.PROMPT_PATH,
        "ground_truth_path": GROUND_TRUTH_PATH,
        "evaluated_records": evaluated,
        "failed_records": len(data) - evaluated,
        "results": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---- 打印 ----
    def pct(x):
        return f"{x*100:6.2f}"

    print(f"\nEvaluated {evaluated} records (failed: {len(data)-evaluated})")
    print("Tolerant scoring: 多预测不罚（只要 gt ⊆ pred）\n")
    header = f"{'subset':<22}{'N':>6}   {'Precision':>10} {'Recall':>9} {'F1':>9}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in SUBSETS:
        r = results[s]
        print(f"{s:<22}{r['n']:>6}   "
              f"{pct(r['precision'])}% {pct(r['recall'])}% {pct(r['f1'])}%")
    print("=" * len(header))
    print(f"\nMetrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
