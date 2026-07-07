# Running in Docker

This repo is packaged so the product-label and hazard-label/category pipeline installs
and runs cleanly anywhere — no manual dependency setup. Code, prompts and data are all
included.

## What runs
| Mode | Script | Purpose | LLM? |
|------|--------|---------|------|
| `test-product` (default) | `evaluate_product_label.py` | product_label vs 2024 ground truth (Precision/Recall/F1) | ✅ Bedrock |
| `test-hazard` | `hazard_eval_gt2024.py` | hazard label/category vs 2024 ground truth | ❌  table lookup |
| `predict-product` | `Claude_eval.py` | predict product_label on the 2024 batch file (can change)| ✅ Bedrock |
| `predict-hazard` | `hazard_eval.py` | predict hazard label/category on the 2024 batch file (can change) | ❌  table lookup |

## Prerequisites
- **Docker Desktop** installed and running (`docker info` should succeed).
- PyCharm ↔ Docker: https://www.jetbrains.com/help/pycharm/docker.html#connect_to_docker

## Build
```bash
docker build -t rasff_labels .
```

## Run
```bash
docker run --rm rasff_labels                          # 2024 product-label test (full)
docker run --rm rasff_labels test-product --limit 20  # quick product test (20 records)
docker run --rm rasff_labels test-hazard              # 2024 hazard test
docker run --rm rasff_labels predict-product          # product predict on 2024 batch
docker run --rm rasff_labels predict-hazard           # hazard predict (table-lookup only, no token needed)
```

### Bedrock credentials
The scripts fall back to a baked-in `AWS_BEARER_TOKEN_BEDROCK`, but it can expire — prefer
passing your own:
```bash
docker run --rm -e AWS_BEARER_TOKEN_BEDROCK="<your-token>" rasff_labels
```
Region `ap-southeast-1`, model `global.anthropic.claude-sonnet-4-6`. `predict-hazard` is
table-lookup only and needs no token.

### Cheap smoke testing
```bash
docker run --rm rasff_labels test-product --limit 5      # 5 LLM calls
docker run --rm -e DEBUG_N=5 rasff_labels predict-product
```

### Getting results out
Outputs are written under `Assay_attr_Extraction_Codes/` (e.g. `eval_product_label_output/`,
`Outputs_hazard_eval_gt2024/`, `Outputs_predict_batch_files/`). Mount a volume to keep them:
```bash
docker run --rm -v "$PWD/out:/app/Assay_attr_Extraction_Codes/eval_product_label_output" rasff_labels
```

## Running locally (without Docker)
The scripts auto-detect the repo root, so from the repo directory:
```bash
pip install -r Assay_attr_Extraction_Codes/requirements.txt
python Assay_attr_Extraction_Codes/evaluate_product_label.py --limit 20
python Assay_attr_Extraction_Codes/hazard_eval.py
```

## Layout
```
.
├── Assay_attr_Extraction_Codes/   # the 4 scripts + imported helpers + prompts/config + requirements
├── FIND-food-recall-data-main_V2/ # product_labels.txt + 2024 batch files
├── hazard/                        # hazard gold + mapping json
├── rasff_2024_ground_truth_labels.json  # 2024 ground truth (used by the two test modes)
├── Dockerfile
└── .dockerignore
```
All paths derive from `RASFF_ROOT` (the repo root locally; `/app` in the image).
