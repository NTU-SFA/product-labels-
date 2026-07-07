import os
import json
import time
import glob
from typing import List, Dict, Any
from string import Formatter

from tqdm import tqdm
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import PromptTemplate

# =========================
# Bedrock auth & model config
# =========================
os.environ.setdefault(
    "AWS_BEARER_TOKEN_BEDROCK",
    "REPLACE_WITH_YOUR_AWS_BEARER_TOKEN_BEDROCK",
)

AWS_REGION = "ap-southeast-1"
CLAUDE_MODEL = "global.anthropic.claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 1024

# 项目根目录：本机默认用绝对路径；在 Docker 里用 RASFF_ROOT=/app 覆盖即可，代码零改动。
RASFF_ROOT = os.environ.get("RASFF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.join(RASFF_ROOT, "FIND-food-recall-data-main_V2")
PROMPT_BASE_DIR = os.path.join(RASFF_ROOT, "Assay_attr_Extraction_Codes")

# 现在只用这一个全集标签文件
PRODUCT_LABELS_PATH = os.path.join(BASE_DIR, "product_labels.txt")

PROMPT_PATH = os.path.join(PROMPT_BASE_DIR, "product_label_prompt_V3.txt")

# =========================
# 这里改成你要处理的"文件夹"，脚本会处理里面所有 .json 文件
# 可用 INPUT_DIR 环境变量覆盖。
# =========================
INPUT_DIR = os.environ.get(
    "INPUT_DIR",
    os.path.join(PROMPT_BASE_DIR, "rasff_information_extraction_2024_V2_todo"),
)


# 处理结果输出目录（每个输入文件对应一个同名输出文件）
OUTPUT_DIR = "./rasff_information_extraction_2024_V2_product"
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "prediction_summary_folder.json")

# 是否把每个文件原地覆盖（True = 直接改输入文件；False = 写到 OUTPUT_DIR）
OVERWRITE_IN_PLACE = False

# True 时同时保留一个 predicted_product_label 字段（便于排查）；
# 无论如何都会把预测写进 product_label
KEEP_PREDICTED_FIELD = False

# True 时跳过 OUTPUT_DIR 里已经生成过输出的文件（用于断点续跑/只补失败的）
# 注意：OVERWRITE_IN_PLACE=True 时此项无意义
SKIP_EXISTING = True

# True 时只处理"还有非空 error 文件"的输入（用于专门重跑之前失败的那些）
ONLY_RERUN_ERRORS = False

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
        pass

    # 健壮兜底：从第一个 '{' 开始用 raw_decode 只取第一个合法 JSON 对象，
    # 自动忽略尾随的多余内容（模型有时会输出多个对象或附带文字）
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
    prompt_variables = extract_template_variables(prompt_text)

    prompt = PromptTemplate(
        input_variables=prompt_variables,
        template=prompt_text
    )

    llm = ChatBedrockConverse(
        model=CLAUDE_MODEL,
        region_name=AWS_REGION,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return prompt | llm, prompt_variables


def predict_one_record(
    item: Dict[str, Any],
    chain,
    prompt_variables: List[str],
    used_labels_str: str,
    extra_labels_str: str,
    available_labels_str: str,
    available_set: set,
) -> List[str]:
    # 兼容新旧数据：旧数据用 notification_type，新数据用 product_type
    product_type_value = (
        safe_get(item, "notification_type")
        or safe_get(item, "product_type")
    )

    full_payload = {
        "used_candidate_labels": used_labels_str,
        "additional_candidate_labels": extra_labels_str,
        "available_candidate_labels": available_labels_str,
        "notification_type": product_type_value,
        "product_type": product_type_value,
        "product_category": safe_get(item, "product_category"),
        "product": safe_get(item, "product"),
        "subject": safe_get(item, "subject"),
    }

    payload = {k: full_payload[k] for k in prompt_variables if k in full_payload}

    response = chain.invoke(payload)
    raw_output = str(response.content)
    parsed = parse_json_output(raw_output)

    # 兼容模型直接返回一个 JSON 数组（如 [] 或 ["sesame seed"]）而不是
    # {"product_label": [...]} 对象的情况：此时整个数组就是标签列表。
    if isinstance(parsed, list):
        label_value = parsed
    else:
        label_value = parsed.get("product_label", [])

    pred = normalize_labels(label_value)
    pred = [x for x in pred if x in available_set]
    return pred


# =========================
# Prediction for one file
# =========================
def predict_one_file(
    input_fp: str,
    chain,
    prompt_variables: List[str],
    used_labels_str: str,
    extra_labels_str: str,
    available_labels_str: str,
    available_set: set,
) -> Dict[str, Any]:
    data = load_json(input_fp)

    # 兼容两种结构：整个文件是一个 list[record]，或单个 record(dict)
    single_record = False
    if isinstance(data, dict):
        data = [data]
        single_record = True

    if DEBUG_N is not None:
        data = data[:DEBUG_N]

    results = []
    errors = []

    for idx, item in enumerate(tqdm(data, desc=f"Predicting {os.path.basename(input_fp)}")):
        time.sleep(0.01)

        out_item = dict(item)
        try:
            pred = predict_one_record(
                item, chain, prompt_variables,
                used_labels_str, extra_labels_str, available_labels_str, available_set,
            )

            # 关键：把预测写进已有的 product_label 字段
            out_item["product_label"] = pred
            if KEEP_PREDICTED_FIELD:
                out_item["predicted_product_label"] = pred
                out_item["prediction_source"] = "llm"
                out_item["model"] = CLAUDE_MODEL

            results.append(out_item)

            if idx < 3:
                print("\n[Sample Prediction]")
                print(json.dumps({
                    "file": os.path.basename(input_fp),
                    "reference": item.get("reference", ""),
                    "product": item.get("product", ""),
                    "subject": item.get("subject", ""),
                    "product_label": pred,
                }, ensure_ascii=False, indent=2))

        except Exception as e:
            # 出错时保留原记录（product_label 维持原样，通常是 []），并记录错误
            results.append(out_item)
            err = {
                "file": os.path.basename(input_fp),
                "reference": item.get("reference", ""),
                "product": item.get("product", ""),
                "subject": item.get("subject", ""),
                "error": str(e),
            }
            errors.append(err)

            if len(errors) <= 5:
                print("\n[Sample Error]")
                print(json.dumps(err, ensure_ascii=False, indent=2))

    # 决定输出路径：文件保持原始名字，写到 OUTPUT_DIR（文件夹名才带 _product）
    if OVERWRITE_IN_PLACE:
        pred_output_fp = input_fp
    else:
        pred_output_fp = os.path.join(OUTPUT_DIR, os.path.basename(input_fp))

    err_output_fp = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(os.path.basename(input_fp))[0] + "_prediction_errors.json",
    )

    # 写回时保持原文件结构（单 record 就写回 dict）
    payload_to_write = results[0] if (single_record and results) else results

    with open(pred_output_fp, "w", encoding="utf-8") as f:
        json.dump(payload_to_write, f, indent=2, ensure_ascii=False)

    # 只有真的有错误时才写 error 文件；没有错误就不生成
    if errors:
        with open(err_output_fp, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    elif os.path.exists(err_output_fp):
        os.remove(err_output_fp)

    return {
        "input_file": input_fp,
        "output_file": pred_output_fp,
        "error_file": err_output_fp if errors else None,
        "input_records": len(data),
        "successful_predictions": len(results) - len(errors),
        "errors": len(errors),
    }


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        raise ValueError("Please set a valid AWS_BEARER_TOKEN_BEDROCK.")

    # 只用一个全集标签文件
    available_labels = normalize_labels(load_labels(PRODUCT_LABELS_PATH))
    available_set = set(available_labels)

    # prompt 仍有三个占位符：全集填给 used/available，additional 给空
    used_labels_str = json.dumps(available_labels, ensure_ascii=False)
    extra_labels_str = json.dumps([], ensure_ascii=False)
    available_labels_str = json.dumps(available_labels, ensure_ascii=False)

    chain, prompt_variables = build_chain(PROMPT_PATH)

    # 收集文件夹里所有 .json（排除我们自己写出的 *_prediction_errors.json）
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    input_files = [fp for fp in input_files if not fp.endswith("_prediction_errors.json")]

    if not input_files:
        print(f"[Warn] No .json files found in: {INPUT_DIR}")
        return

    total_found = len(input_files)

    # 只重跑还留着非空 error 文件的那些输入
    if ONLY_RERUN_ERRORS and not OVERWRITE_IN_PLACE:
        rerun_stems = set()
        for ef in glob.glob(os.path.join(OUTPUT_DIR, "*_prediction_errors.json")):
            try:
                with open(ef, "r", encoding="utf-8") as f:
                    if json.load(f):  # 非空列表才算需要重跑
                        rerun_stems.add(os.path.basename(ef).replace("_prediction_errors.json", ""))
            except Exception:
                pass
        input_files = [
            fp for fp in input_files
            if os.path.splitext(os.path.basename(fp))[0] in rerun_stems
        ]
        print(f"[ONLY_RERUN_ERRORS] {len(input_files)} file(s) to re-run")

    # 跳过已经生成过输出的文件（断点续跑）
    elif SKIP_EXISTING and not OVERWRITE_IN_PLACE:
        input_files = [
            fp for fp in input_files
            if not os.path.exists(os.path.join(OUTPUT_DIR, os.path.basename(fp)))
        ]
        print(f"[SKIP_EXISTING] {total_found - len(input_files)} already done, {len(input_files)} remaining")

    if not input_files:
        print("[Info] Nothing to process (all done / no matching files).")
        return

    print(f"Found {total_found} json file(s) in {INPUT_DIR}; processing {len(input_files)}")

    file_summaries = []
    for input_fp in input_files:
        summary = predict_one_file(
            input_fp=input_fp,
            chain=chain,
            prompt_variables=prompt_variables,
            used_labels_str=used_labels_str,
            extra_labels_str=extra_labels_str,
            available_labels_str=available_labels_str,
            available_set=available_set,
        )
        summary["status"] = "done"
        file_summaries.append(summary)

    overall = {
        "model": CLAUDE_MODEL,
        "product_labels_path": PRODUCT_LABELS_PATH,
        "num_labels": len(available_set),
        "prompt_path": PROMPT_PATH,
        "prompt_variables": prompt_variables,
        "input_dir": INPUT_DIR,
        "overwrite_in_place": OVERWRITE_IN_PLACE,
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
