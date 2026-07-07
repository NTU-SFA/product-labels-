# Product and Hazard

Product-label extraction and hazard attribute/labelling for RASFF food-recall notifications,
using Claude on Amazon Bedrock (product records) plus a table-lookup engine (hazard records).

## Pipeline

**Product-label extraction** — prompt: `Assay_attr_Extraction_Codes/product_label_prompt_V3.txt`
- test on 2024 ground truth (Precision/Recall/F1): **`Assay_attr_Extraction_Codes/evaluate_product_label.py`**
- predict on the 2024 batch files: **`Assay_attr_Extraction_Codes/Claude_eval.py`**

**Hazard attribute extraction & labelling**
- test on 2024 ground truth (table-lookup): **`Assay_attr_Extraction_Codes/hazard_eval_gt2024.py`**
- predict on the 2024 batch files (table-lookup): **`Assay_attr_Extraction_Codes/hazard_eval.py`**

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
├── Assay_attr_Extraction_Codes/   # 4 pipeline scripts + imported helpers + prompts/config + requirements
├── FIND-food-recall-data-main_V2/ # product_labels.txt + 2024 batch files
├── hazard/                        # hazard gold + mapping JSON
├── rasff_2024_ground_truth_labels.json   # 2024 ground truth (used by the two test scripts)
├── Dockerfile, .dockerignore, README_docker.md
└── (example result JSONs for the first batch kept at the root)
```

All paths derive from `RASFF_ROOT` (the repo root locally; `/app` inside the Docker image).
