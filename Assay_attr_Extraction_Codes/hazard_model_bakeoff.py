"""在 2024 GT 的 no_hazards 记录上横评多个 Bedrock 模型（Claude 被账号门禁时的替补选型）。

固定同一份抽样子集 + 同一份 prompt（默认跟随主 pipeline），逐模型打分：
  exact       整条记录标签集合完全一致的比例（主指标，和正式评估同定义）
  jaccard     平均 Jaccard
  P/R/F1      micro
  bad_json    输出解析失败的比例（本任务的硬门槛）
  raw_kept    原始输出里能落进 1021 词表的比例（词表遵从度）

用法：
    python3 hazard_model_bakeoff.py                       # 默认 60 条 × 内置候选
    SAMPLE_N=300 MODELS=zai.glm-5,deepseek.v3.2 python3 hazard_model_bakeoff.py
    SAMPLE_N=300 MODELS=zai.glm-5 REGION=us-east-1 python3 hazard_model_bakeoff.py
"""
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

import hazard_predict_folder_llm as L

GT_PATH = os.environ.get("GROUND_TRUTH_PATH",
                         os.path.join(L.RASFF_ROOT, "rasff_2024_ground_truth_labels.json"))
PROMPT_PATH = os.environ.get("PROMPT_PATH", L.NO_HAZARD_PROMPT_PATH)
OUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(L.CODE_DIR, "Outputs_hazard_model_bakeoff"))
SAMPLE_N = int(os.environ.get("SAMPLE_N", "60"))
WORKERS = int(os.environ.get("WORKERS", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
APPLY_POST_PROCESS = os.environ.get("APPLY_POST_PROCESS", "1") == "1"
PRE_CANON = os.environ.get("PRE_CANON", "1") == "1"

# (region, modelId) —— 均已实测 converse 可调通且能守住 JSON-only
DEFAULT_MODELS = [
    ("us-east-1", "zai.glm-5"),
    ("us-east-1", "moonshotai.kimi-k2.5"),
    ("us-east-1", "deepseek.v3.2"),
    ("us-east-1", "mistral.mistral-large-3-675b-instruct"),
    ("us-east-1", "qwen.qwen3-next-80b-a3b"),
    ("us-east-1", "us.meta.llama4-maverick-17b-instruct-v1:0"),
    ("us-east-1", "nvidia.nemotron-super-3-120b"),
    ("us-east-1", "us.amazon.nova-pro-v1:0"),
    ("us-east-1", "us.writer.palmyra-x5-v1:0"),
]


def models_from_env():
    spec = os.environ.get("MODELS", "").strip()
    if not spec:
        return DEFAULT_MODELS
    region = os.environ.get("REGION", "us-east-1")
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(tuple(item.split("@", 1)[::-1]) if "@" in item else (region, item))
    return out


def norm_set(labels):
    return {x.strip().lower() for x in labels if isinstance(x, str) and x.strip()}


class Metric:
    def __init__(self):
        self.n = self.exact = self.bad = 0
        self.jac = 0.0
        self.tp = self.fp = self.fn = 0
        self.raw_total = self.raw_kept = 0

    def add(self, pred, truth):
        self.n += 1
        if pred == truth:
            self.exact += 1
        inter, union = pred & truth, pred | truth
        self.jac += (len(inter) / len(union)) if union else 1.0
        self.tp += len(inter); self.fp += len(pred - truth); self.fn += len(truth - pred)

    def report(self, secs):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return {"n": self.n,
                "exact": round(self.exact / self.n, 4) if self.n else 0.0,
                "jaccard": round(self.jac / self.n, 4) if self.n else 0.0,
                "micro_p": round(p, 4), "micro_r": round(r, 4),
                "micro_f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
                "bad_json": round(self.bad / self.n, 4) if self.n else 0.0,
                "raw_kept": round(self.raw_kept / self.raw_total, 4) if self.raw_total else None,
                "secs": round(secs, 1)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    label_list, label_lookup, label_to_cat = L.build_label_space()
    L.build_canon(label_list)
    labels_str = json.dumps(label_list, ensure_ascii=False)
    prompt_text = L.load_text(PROMPT_PATH)
    prompt_vars = L.extract_template_variables(prompt_text)

    gt = L.load_json(GT_PATH)
    no_recs = [r for r in gt if not (isinstance(r.get("hazards"), list) and r.get("hazards"))]
    no_recs.sort(key=lambda r: str(r.get("reference", "")))          # 抽样确定化
    step = max(1, len(no_recs) // SAMPLE_N)
    sample = no_recs[::step][:SAMPLE_N]
    print(f"no_hazards={len(no_recs)} -> sample={len(sample)} | prompt={os.path.basename(PROMPT_PATH)} "
          f"| label space={len(label_list)}")

    def render(r):
        ptype = L.safe_get(r, "product_type") or L.safe_get(r, "notification_type")
        payload = {"allowed_hazard_labels": labels_str, "product_type": ptype,
                   "notification_type": ptype,
                   "product_category": L.safe_get(r, "product_category"),
                   "product": L.safe_get(r, "product"), "subject": L.safe_get(r, "subject")}
        return prompt_text.format(**{k: payload.get(k, "") for k in prompt_vars})

    prompts = {str(r.get("reference", "")): render(r) for r in sample}

    def score(raw_by_ref, metric):
        for r in sample:
            raw = raw_by_ref.get(str(r.get("reference", "")))
            if raw is None:
                metric.bad += 1
                raw = []
            metric.raw_total += len(raw)
            labels = L.canon_list(raw) if PRE_CANON else raw
            labels = L.canon_list(L.canon_filter(labels, label_lookup))
            metric.raw_kept += len(labels)
            if APPLY_POST_PROCESS:
                labels = L.post_process_labels(labels)
            metric.add(norm_set(labels), norm_set(r.get("hazard_label") or []))
        return metric

    results, raw_dump = {}, {}

    # 基线：旧 Claude Sonnet 4.5 + V1 prompt 的缓存输出（同一子集，供对照）
    cache_path = os.path.join(L.CODE_DIR, "Outputs_hazard_eval_gt2024_spec", "llm_raw_cache.json")
    if os.path.exists(cache_path):
        cache = L.load_json(cache_path)
        base = {k: (None if isinstance(v, dict) else v)
                for k, v in cache.items() if k in prompts}
        if base:
            results["[baseline] claude-sonnet-4.5 + promptV1 (cached)"] = score(base, Metric()).report(0.0)

    cfg = Config(retries={"max_attempts": 4, "mode": "adaptive"},
                 read_timeout=180, connect_timeout=15)

    for region, model in models_from_env():
        rt = boto3.client("bedrock-runtime", region_name=region, config=cfg)

        def call(ref):
            try:
                resp = rt.converse(modelId=model,
                                   messages=[{"role": "user",
                                              "content": [{"text": prompts[ref]}]}],
                                   inferenceConfig={"temperature": 0, "maxTokens": MAX_TOKENS})
                txt = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
                parsed = L.parse_json_output(str(txt))
                if isinstance(parsed, list):
                    parsed = {"hazard_label": parsed}
                out = parsed.get("hazard_label", [])
                return ref, ([x for x in out if isinstance(x, str)] if isinstance(out, list) else None)
            except Exception as e:
                return ref, ("__ERR__" + f"{type(e).__name__}: {e}"[:120])

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            pairs = list(ex.map(call, list(prompts)))
        secs = time.time() - t0

        errs = [v for _, v in pairs if isinstance(v, str)]
        raw_by_ref = {k: v for k, v in pairs if isinstance(v, list)}
        m = score(raw_by_ref, Metric())
        key = f"{model} @ {region}"
        results[key] = m.report(secs)
        results[key]["api_errors"] = len(errs)
        results[key]["error_sample"] = errs[0][7:] if errs else ""
        raw_dump[key] = raw_by_ref
        rep = results[key]
        print(f"  {key:52} exact={rep['exact']:.3f} jac={rep['jaccard']:.3f} F1={rep['micro_f1']:.3f} "
              f"bad={rep['bad_json']:.2f} kept={rep['raw_kept']} err={len(errs)} {secs:.0f}s", flush=True)
        json.dump(results, open(os.path.join(OUT_DIR, "bakeoff_metrics.json"), "w"),
                  ensure_ascii=False, indent=2)
        json.dump(raw_dump, open(os.path.join(OUT_DIR, "bakeoff_raw.json"), "w"),
                  ensure_ascii=False, indent=1)

    print(f"\n{'model':54} {'exact':>7} {'jac':>7} {'F1':>7} {'badJSON':>8} {'kept':>7} {'err':>4} {'secs':>6}")
    print("-" * 110)
    for k, v in sorted(results.items(), key=lambda z: -z[1]["exact"]):
        print(f"{k:54} {v['exact']:>7.4f} {v['jaccard']:>7.4f} {v['micro_f1']:>7.4f} "
              f"{v['bad_json']:>8.3f} {str(v['raw_kept']):>7} {v.get('api_errors', 0):>4} {v['secs']:>6.0f}")
    print(f"\nSaved: {OUT_DIR}/bakeoff_metrics.json + bakeoff_raw.json")


if __name__ == "__main__":
    main()
