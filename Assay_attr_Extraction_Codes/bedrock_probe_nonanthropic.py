"""探测当前 key 能用的**非 Anthropic** 模型，并顺带测三件事：
  1) converse 能不能调通
  2) 能不能守住 JSON-only 输出格式（本任务硬要求）
  3) 支不支持 cachePoint（prompt caching，决定成本）

用法：
    python3 bedrock_probe_nonanthropic.py
    REGIONS=us-east-1 python3 bedrock_probe_nonanthropic.py
"""
import os
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import boto3

REGIONS = [x.strip() for x in os.environ.get(
    "REGIONS", "us-east-1,eu-central-1,ap-southeast-1").split(",") if x.strip()]
WORKERS = int(os.environ.get("WORKERS", "8"))

# JSON-only 遵从度测试：和真任务同构的迷你版
JSON_PROBE = (
    'Allowed hazard_label values: ["Aflatoxins","Allergens (milk)","Foreign bodies (glass)","Salmonella spp"]\n'
    'Output valid JSON only, no explanation, no markdown fences.\n'
    'Return exactly this format: {"hazard_label":["label1"]}\n\n'
    'subject: Undeclared milk in dark chocolate from Belgium\n'
)


def candidates():
    out = []
    for region in REGIONS:
        c = boto3.client("bedrock", region_name=region)
        seen = set()
        try:
            for p in c.list_inference_profiles(maxResults=100)["inferenceProfileSummaries"]:
                pid = p["inferenceProfileId"]
                if "anthropic" not in pid:
                    out.append((region, pid)); seen.add(pid)
        except Exception as e:
            print(f"[warn] {region} profiles: {type(e).__name__} {e}")
        try:
            for m in c.list_foundation_models()["modelSummaries"]:
                mid = m["modelId"]
                if m.get("providerName", "").lower() == "anthropic" or mid in seen:
                    continue
                if "TEXT" not in (m.get("outputModalities") or []):
                    continue
                if "ON_DEMAND" not in (m.get("inferenceTypesSupported") or []):
                    continue    # 只要按需付费的，PROVISIONED 的跳过
                if ":" in mid.split("-")[-1]:
                    continue    # 跳过 :48k/:200k 之类变体
                out.append((region, mid))
        except Exception as e:
            print(f"[warn] {region} models: {type(e).__name__} {e}")
    return sorted(set(out))


def probe(rm):
    region, model = rm
    rt = boto3.client("bedrock-runtime", region_name=region)
    row = {"region": region, "model": model, "status": "", "json_ok": False,
           "cache_ok": False, "reply": "", "error": ""}
    try:
        r = rt.converse(modelId=model,
                        messages=[{"role": "user", "content": [{"text": JSON_PROBE}]}],
                        inferenceConfig={"temperature": 0, "maxTokens": 64})
        txt = "".join(b.get("text", "") for b in r["output"]["message"]["content"]).strip()
        row["status"] = "OK"
        row["reply"] = txt[:120]
        row["json_ok"] = txt.startswith("{") and txt.endswith("}") and "hazard_label" in txt
    except Exception as e:
        msg = str(e)
        row["status"] = ("GATE" if "use case details" in msg else
                         "NOACCESS" if "AccessDenied" in msg or "not authorized" in msg else
                         "INVALID" if "ValidationException" in type(e).__name__ or "invalid" in msg else
                         type(e).__name__)
        row["error"] = msg[:130]
        return row
    try:                       # cachePoint 支持度（长前缀才有意义，这里只测接受不接受）
        rt.converse(modelId=model,
                    messages=[{"role": "user", "content": [
                        {"text": "x " * 20}, {"cachePoint": {"type": "default"}},
                        {"text": "Say OK"}]}],
                    inferenceConfig={"temperature": 0, "maxTokens": 8})
        row["cache_ok"] = True
    except Exception:
        pass
    return row


def main():
    cands = candidates()
    print(f"probing {len(cands)} non-Anthropic (region, model) combos across {REGIONS} ...")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(probe, cands))

    ok = [r for r in res if r["status"] == "OK"]
    print(f"\n{'='*100}\nUSABLE: {len(ok)} / {len(res)}   (json = 守住纯 JSON 输出, cache = 支持 cachePoint)\n{'='*100}")
    print(f"{'json':5} {'cache':6} {'region':16} model")
    print("-" * 100)
    for r in sorted(ok, key=lambda z: (not z["json_ok"], not z["cache_ok"], z["region"], z["model"])):
        print(f"{'Y' if r['json_ok'] else '.':5} {'Y' if r['cache_ok'] else '.':6} "
              f"{r['region']:16} {r['model']}")
    print(f"\nstatus counts: {dict(Counter(r['status'] for r in res))}")
    for r in sorted(res, key=lambda z: (z["status"], z["region"], z["model"])):
        if r["status"] != "OK":
            print(f"  {r['status']:10} {r['region']:16} {r['model']:60} {r['error'][:60]}")
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "bedrock_probe_nonanthropic_result.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
