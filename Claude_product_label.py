import os
import json
import time
import random
from typing import List, Dict, Any
from string import Formatter

from tqdm import tqdm
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import PromptTemplate

# =========================
# Bedrock auth & model config
# =========================
# Bedrock long-term API key (read from env; fall back to the hardcoded value).
os.environ.setdefault(
    "AWS_BEARER_TOKEN_BEDROCK",
    "xxxx",
)

AWS_REGION = "ap-southeast-1"

# Bedrock model id (note: this is the Bedrock id, not the Anthropic API id)
CLAUDE_MODEL = "global.anthropic.claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 1024

# 每个数据集随机抽 x 条，总共评估 4x 条
SAMPLE_PER_DATASET = 100
RANDOM_SEED = None

# 评估阶段一般建议 False，更公平对齐已有人工标签空间
ALLOW_ADDITIONAL_LABELS = False

BASE_DIR = r"/Users/xueluangong/Desktop/GPT Source Codes/FIND-food-recall-data-main_V2"

PRODUCT_LABELS_IN_LABELLED_PATH = os.path.join(BASE_DIR, "product_labels_in_labelled_data.txt")
PRODUCT_LABELS_NOT_IN_LABELLED_PATH = os.path.join(BASE_DIR, "product_labels_not_in_labelled_data.txt")

SINGLE_LABEL_PATH = os.path.join(BASE_DIR, "rasff_data_with_single_product_label.json")
MULTIPLE_LABEL_PATH = os.path.join(BASE_DIR, "rasff_data_with_multiple_product_labels.json")
ANIMAL_FEED_PATH = os.path.join(BASE_DIR, "rasff_data_with_product_label_animal_feed.json")
FOOD_CONTACT_PATH = os.path.join(BASE_DIR, "rasff_data_with_product_label_food_contact_material.json")

PROMPT_PATH = "/Users/xueluangong/Desktop/GPT Source Codes/Assay_attr_Extraction_Codes/product_label_prompt_V3.txt"

OUTPUT_DIR = "./Outputs_eval_prompt_only"
PREDICTION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "eval_predictions_stratified_prompt_only.json")
ERROR_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "eval_errors_stratified_prompt_only.json")
WRONG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "eval_wrong_cases_stratified_prompt_only.json")
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "eval_summary_stratified_prompt_only.json")


# =========================
# Utilities
# =========================
def load_json(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text(fp: str) -> str:
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()

def load_labels(fp: str) -> List[str]:
    labels = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                labels.append(x)
    return labels

def normalize_labels(labels: List[str]) -> List[str]:
    if not isinstance(labels, list):
        return []
    cleaned = []
    for x in labels:
        if isinstance(x, str):
            x = x.strip().lower()
            if x:
                cleaned.append(x)
    return sorted(list(set(cleaned)))

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

def exact_match(pred: List[str], gold: List[str]) -> int:
    return int(normalize_labels(pred) == normalize_labels(gold))

def safe_get(item: Dict[str, Any], key: str) -> str:
    value = item.get(key, "")
    if value is None:
        return ""
    return str(value)

def extract_template_variables(template_text: str) -> List[str]:
    vars_found = []
    for _, field_name, _, _ in Formatter().parse(template_text):
        if field_name:
            vars_found.append(field_name)
    return sorted(list(set(vars_found)))

def build_chain(prompt_path: str):
    prompt_text = load_text(prompt_path)
    input_variables = extract_template_variables(prompt_text)

    prompt = PromptTemplate(
        input_variables=input_variables,
        template=prompt_text
    )

    llm = ChatBedrockConverse(
        model=CLAUDE_MODEL,
        region_name=AWS_REGION,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return prompt | llm, input_variables


# =========================
# Sampling
# =========================
def sample_dataset(data: List[Dict[str, Any]], sample_size: int, seed: int, dataset_name: str):
    if sample_size > len(data):
        raise ValueError(
            f"SAMPLE_PER_DATASET={sample_size} is larger than dataset size {len(data)} for {dataset_name}"
        )
    rng = random.Random(seed)
    return rng.sample(data, sample_size)

def build_eval_pool(sample_per_dataset: int, seed):
    datasets = {
        "single_label": load_json(SINGLE_LABEL_PATH),
        "multiple_labels": load_json(MULTIPLE_LABEL_PATH),
        "animal_feed": load_json(ANIMAL_FEED_PATH),
        "food_contact_material": load_json(FOOD_CONTACT_PATH),
    }

    eval_pool = []
    for i, (dataset_name, dataset_data) in enumerate(datasets.items()):
        if seed is None:
            sampled = random.sample(dataset_data, sample_per_dataset)
        else:
            sampled = sample_dataset(dataset_data, sample_per_dataset, seed + i, dataset_name)

        for item in sampled:
            new_item = dict(item)
            new_item["_dataset_name"] = dataset_name
            eval_pool.append(new_item)

    return eval_pool


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        raise ValueError("Please set a valid AWS_BEARER_TOKEN_BEDROCK.")

    used_labels = normalize_labels(load_labels(PRODUCT_LABELS_IN_LABELLED_PATH))
    extra_labels = normalize_labels(load_labels(PRODUCT_LABELS_NOT_IN_LABELLED_PATH))

    if ALLOW_ADDITIONAL_LABELS:
        available_labels = sorted(list(set(used_labels + extra_labels)))
    else:
        available_labels = used_labels

    available_set = set(available_labels)

    used_labels_str = json.dumps(used_labels, ensure_ascii=False)
    extra_labels_str = json.dumps(extra_labels if ALLOW_ADDITIONAL_LABELS else [], ensure_ascii=False)
    available_labels_str = json.dumps(available_labels, ensure_ascii=False)

    eval_data = build_eval_pool(SAMPLE_PER_DATASET, RANDOM_SEED)
    chain, prompt_variables = build_chain(PROMPT_PATH)

    results = []
    errors = []
    wrong_cases = []

    total = 0
    correct = 0

    per_dataset_stats = {
        "single_label": {"total": 0, "correct": 0},
        "multiple_labels": {"total": 0, "correct": 0},
        "animal_feed": {"total": 0, "correct": 0},
        "food_contact_material": {"total": 0, "correct": 0},
    }

    for idx, item in enumerate(tqdm(eval_data, desc="Prompt-only stratified evaluation")):
        time.sleep(0.01)
        dataset_name = item["_dataset_name"]

        try:
            full_payload = {
                "used_candidate_labels": used_labels_str,
                "additional_candidate_labels": extra_labels_str,
                "available_candidate_labels": available_labels_str,
                # 支持新旧 prompt 两种变量名
                "notification_type": safe_get(item, "notification_type"),
                "product_type": safe_get(item, "notification_type"),
                "product_category": safe_get(item, "product_category"),
                "product": safe_get(item, "product"),
                "subject": safe_get(item, "subject"),
            }

            payload = {k: full_payload[k] for k in prompt_variables if k in full_payload}

            response = chain.invoke(payload)

            raw_output = str(response.content)
            parsed = parse_json_output(raw_output)

            pred = normalize_labels(parsed.get("product_label", []))
            pred = [x for x in pred if x in available_set]

            gold = normalize_labels(item.get("product_label", []))
            match = exact_match(pred, gold)

            total += 1
            correct += match
            per_dataset_stats[dataset_name]["total"] += 1
            per_dataset_stats[dataset_name]["correct"] += match

            out_item = dict(item)
            out_item["predicted_product_label"] = pred
            out_item["gold_product_label"] = gold
            out_item["exact_match"] = match
            out_item["prediction_source"] = "llm"
            out_item["model"] = CLAUDE_MODEL
            out_item["raw_output"] = raw_output
            results.append(out_item)

            if match == 0:
                wrong_cases.append(out_item)

            if idx < 3:
                print("\n[Sample Prediction]")
                print(json.dumps({
                    "dataset": dataset_name,
                    "reference": item.get("reference", ""),
                    "product": item.get("product", ""),
                    "subject": item.get("subject", ""),
                    "gold": gold,
                    "pred": pred,
                    "match": match
                }, ensure_ascii=False, indent=2))

        except Exception as e:
            err = {
                "dataset": dataset_name,
                "reference": item.get("reference", ""),
                "product": item.get("product", ""),
                "subject": item.get("subject", ""),
                "error": str(e)
            }
            errors.append(err)

            if len(errors) <= 5:
                print("\n[Sample Error]")
                print(json.dumps(err, ensure_ascii=False, indent=2))

    overall_accuracy = correct / total if total > 0 else 0.0

    summary = {
        "model": CLAUDE_MODEL,
        "sample_per_dataset": SAMPLE_PER_DATASET,
        "random_seed": RANDOM_SEED,
        "allow_additional_labels": ALLOW_ADDITIONAL_LABELS,
        "prompt_path": PROMPT_PATH,
        "prompt_variables": prompt_variables,
        "overall": {
            "total": total,
            "correct": correct,
            "exact_match_accuracy": overall_accuracy,
            "errors": len(errors),
        },
        "per_dataset": {}
    }

    for dataset_name, stats in per_dataset_stats.items():
        ds_total = stats["total"]
        ds_correct = stats["correct"]
        ds_acc = ds_correct / ds_total if ds_total > 0 else 0.0
        summary["per_dataset"][dataset_name] = {
            "total": ds_total,
            "correct": ds_correct,
            "exact_match_accuracy": ds_acc,
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

    print("\n===== Per-Dataset Evaluation =====")
    for dataset_name, stats in summary["per_dataset"].items():
        print(f"\n[{dataset_name}]")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    print(f"\nWrong cases saved to: {WRONG_OUTPUT_PATH}")
    print(f"Detailed summary saved to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()