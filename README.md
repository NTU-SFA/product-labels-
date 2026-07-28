# Product and Hazard

Product-label extraction and hazard attribute/labelling for RASFF food-recall notifications,
using Claude on Amazon Bedrock (product records) plus a table-lookup engine (hazard records).

## Pipeline

**Product-label extraction** — prompt: `Assay_attr_Extraction_Codes/product_label_prompt_V3.txt`
- test on 2024 ground truth (Precision/Recall/F1): **`Assay_attr_Extraction_Codes/evaluate_product_label.py`**
- predict on the 2024 batch files (can change): **`Assay_attr_Extraction_Codes/Claude_eval.py`**

**Hazard attribute extraction & labelling**

Records that carry a structured `hazards` field are labelled by table lookup; records that
only carry a subject need the no-hazards branch, which is available as keyword rules or as an
LLM. The three scripts differ only in that branch:

| Script | `hazards` field present | subject only | Bedrock token |
|---|---|---|---|
| `Assay_attr_Extraction_Codes/hazard_eval_gt2024.py` — test on 2024 ground truth | table lookup | n-gram keyword rules | not needed |
| `Assay_attr_Extraction_Codes/hazard_eval.py` — predict on the 2024 batch files (can change) | table lookup | n-gram keyword rules | not needed |
| `Assay_attr_Extraction_Codes/hazard_predict_folder_llm.py` — predict hazard on a folder | table lookup | Claude on Bedrock | required |

The table lookup resolves `hazard_label` from `hazard/has_hazards_mapping_hazard_label.json`
(an empty value in that file means the label is the key itself); `hazard_category_label` is
always derived from `hazard_label` through
`hazard/mapping_hazard_label_to_hazard_category_label.json`.

## Run with Docker (recommended)

Everything (code, prompts, data) is bundled, so it installs and runs anywhere — no manual
dependency setup. See **[`README_docker.md`](README_docker.md)** for full details.

```bash
docker build -t rasff_labels .

docker run --rm rasff_labels                          # 2024 product-label test (default)
docker run --rm rasff_labels test-product --limit 20  # quick product test
docker run --rm rasff_labels test-hazard              # 2024 hazard test
docker run --rm rasff_labels predict-product          # product predict on 2024 batch (can change)
docker run --rm rasff_labels predict-hazard           # hazard predict (table-lookup) (can change)
docker run --rm rasff_labels predict-hazard-llm       # hazard predict, no-hazards branch via LLM
```

Bedrock credentials are **not** committed. Pass your token at run time (region `ap-southeast-1`,
model `global.anthropic.claude-sonnet-4-6`):

```bash
docker run --rm -e AWS_BEARER_TOKEN_BEDROCK="<your-token>" rasff_labels
```

## Run locally (without Docker)

The scripts auto-detect the repo root, so from the repo directory:

```bash
pip install -r Assay_attr_Extraction_Codes/requirements.txt
export AWS_BEARER_TOKEN_BEDROCK="<your-token>"   # not needed for hazard_eval.py (table-lookup)

python Assay_attr_Extraction_Codes/evaluate_product_label.py --limit 20
python Assay_attr_Extraction_Codes/hazard_eval.py
```

## Layout

```
.
├── Assay_attr_Extraction_Codes/   # 5 entry scripts + Claude_predict_folder.py (imported by
│                                  # evaluate_product_label.py) + prompts/config + requirements
├── FIND-food-recall-data-main_V2/ # product_labels.txt + 2024 batch files
├── hazard/                        # hazard gold + mapping JSON
├── rasff_2024_ground_truth_labels.json   # 2024 ground truth (used by the two test scripts)
├── Dockerfile, .dockerignore, README_docker.md
└── (example result JSONs for the first batch kept at the root)
```

All paths derive from `RASFF_ROOT` (the repo root locally; `/app` inside the Docker image).
