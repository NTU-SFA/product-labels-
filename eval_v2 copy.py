import os
import json
import time
from typing import List, Dict, Any

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"
TEMPERATURE = 0

# 预测阶段建议 True，这样可以使用 additional labels
ALLOW_ADDITIONAL_LABELS = True

BASE_DIR = r"/Users/xueluangong/Desktop/GPT Source Codes/FIND-food-recall-data-main_V2"
PROMPT_BASE_DIR = r"/Users/xueluangong/Desktop/GPT Source Codes/Assay_attr_Extraction_Codes"

PRODUCT_LABELS_IN_LABELLED_PATH = os.path.join(BASE_DIR, "product_labels_in_labelled_data.txt")
PRODUCT_LABELS_NOT_IN_LABELLED_PATH = os.path.join(BASE_DIR, "product_labels_not_in_labelled_data.txt")

PROMPT_PATH = os.path.join(PROMPT_BASE_DIR, "Food_label_prompt_v2.txt")

# 这里改成你要预测的 6 个文件
INPUT_FILE_PATHS = [
    os.path.join(BASE_DIR, "rasff_data_2024_batch1.json"),
    #os.path.join(BASE_DIR, "rasff_data_2024_batch2.json"),
    #os.path.join(BASE_DIR, "rasff_data_2024_batch3.json"),
    #os.path.join(BASE_DIR, "rasff_data_2024_batch4.json"),
    #os.path.join(BASE_DIR, "rasff_data_2024_batch5.json"),
    #os.path.join(BASE_DIR, "rasff_data_2024_batch6.json"),
]

OUTPUT_DIR = "./Outputs_predict_6files"
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "prediction_summary_6files.json")

# 调试时可以设成 20 / 100；正式跑全量时设为 None
DEBUG_N = None


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
        raise ValueError(f"Cannot parse JSON output: {text}")

def safe_get(item: Dict[str, Any], key: str) -> str:
    value = item.get(key, "")
    if value is None:
        return ""
    return str(value)

def build_chain(prompt_path: str):
    prompt_text = load_text(prompt_path)
    prompt = PromptTemplate(
        input_variables=[
            "used_candidate_labels",
            "additional_candidate_labels",
            "available_candidate_labels",
            "notification_type",
            "product_category",
            "product",
            "subject"
        ],
        template=prompt_text
    )

    llm = ChatOpenAI(
        model_name=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE
    )
    return prompt | llm

def output_path_for_input(input_fp: str) -> str:
    base = os.path.basename(input_fp)
    stem, ext = os.path.splitext(base)
    return os.path.join(OUTPUT_DIR, f"{stem}_with_predicted_product_label{ext}")

def error_path_for_input(input_fp: str) -> str:
    base = os.path.basename(input_fp)
    stem, _ = os.path.splitext(base)
    return os.path.join(OUTPUT_DIR, f"{stem}_prediction_errors.json")


# =========================
# Prediction for one file
# =========================
def predict_one_file(
    input_fp: str,
    chain,
    used_labels_str: str,
    extra_labels_str: str,
    available_labels_str: str,
    available_set: set
) -> Dict[str, Any]:
    data = load_json(input_fp)

    if DEBUG_N is not None:
        data = data[:DEBUG_N]

    results = []
    errors = []

    for idx, item in enumerate(tqdm(data, desc=f"Predicting {os.path.basename(input_fp)}")):
        time.sleep(0.01)

        try:
            response = chain.invoke({
                "used_candidate_labels": used_labels_str,
                "additional_candidate_labels": extra_labels_str,
                "available_candidate_labels": available_labels_str,
                "notification_type": safe_get(item, "notification_type"),
                "product_category": safe_get(item, "product_category"),
                "product": safe_get(item, "product"),
                "subject": safe_get(item, "subject"),
            })

            raw_output = str(response.content)
            parsed = parse_json_output(raw_output)

            pred = normalize_labels(parsed.get("product_label", []))
            pred = [x for x in pred if x in available_set]

            out_item = dict(item)
            out_item["predicted_product_label"] = pred
            out_item["prediction_source"] = "llm"
            #out_item["raw_output"] = raw_output

            results.append(out_item)

            if idx < 3:
                print("\n[Sample Prediction]")
                print(json.dumps({
                    "file": os.path.basename(input_fp),
                    "reference": item.get("reference", ""),
                    "product": item.get("product", ""),
                    "subject": item.get("subject", ""),
                    "predicted_product_label": pred
                }, ensure_ascii=False, indent=2))

        except Exception as e:
            err = {
                "file": os.path.basename(input_fp),
                "reference": item.get("reference", ""),
                "product": item.get("product", ""),
                "subject": item.get("subject", ""),
                "error": str(e)
            }
            errors.append(err)

            if len(errors) <= 5:
                print("\n[Sample Error]")
                print(json.dumps(err, ensure_ascii=False, indent=2))

    # save
    pred_output_fp = output_path_for_input(input_fp)
    err_output_fp = error_path_for_input(input_fp)

    with open(pred_output_fp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(err_output_fp, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    return {
        "input_file": input_fp,
        "output_file": pred_output_fp,
        "error_file": err_output_fp,
        "input_records": len(data),
        "successful_predictions": len(results),
        "errors": len(errors),
    }


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        raise ValueError("Please set a valid OPENAI_API_KEY.")

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

    chain = build_chain(PROMPT_PATH)

    file_summaries = []

    for input_fp in INPUT_FILE_PATHS:
        if not os.path.exists(input_fp):
            print(f"\n[Skip] File not found: {input_fp}")
            file_summaries.append({
                "input_file": input_fp,
                "output_file": None,
                "error_file": None,
                "input_records": 0,
                "successful_predictions": 0,
                "errors": 0,
                "status": "file_not_found"
            })
            continue

        summary = predict_one_file(
            input_fp=input_fp,
            chain=chain,
            used_labels_str=used_labels_str,
            extra_labels_str=extra_labels_str,
            available_labels_str=available_labels_str,
            available_set=available_set
        )
        summary["status"] = "done"
        file_summaries.append(summary)

    overall = {
        "allow_additional_labels": ALLOW_ADDITIONAL_LABELS,
        "model": OPENAI_MODEL,
        "files": file_summaries,
        "total_input_records": sum(x["input_records"] for x in file_summaries),
        "total_successful_predictions": sum(x["successful_predictions"] for x in file_summaries),
        "total_errors": sum(x["errors"] for x in file_summaries),
    }

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n===== Prediction Summary =====")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"\nDetailed summary saved to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()