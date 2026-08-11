# Product and Hazard

Product-label extraction and hazard attribute/labelling for RASFF food-recall notifications,
using Claude on Amazon Bedrock (product records) plus a table-lookup engine (hazard records).

## Pipeline

**Product-label extraction** — prompt: `Assay_attr_Extraction_Codes/product_label_prompt_V3.txt`

| Script | What it does | Input | Output dir | Bedrock token |
|---|---|---|---|---|
| `Assay_attr_Extraction_Codes/evaluate_product_label.py` | test on 2024 ground truth (Precision/Recall/F1) | `rasff_2024_ground_truth_labels.json` (`GROUND_TRUTH_PATH`) | `eval_product_label_output/` | required |
| `Assay_attr_Extraction_Codes/Claude_eval.py` | predict on a 2024 batch file (can change) | `FIND-food-recall-data-main_V2/rasff_data_2024_batch1.json` (batch2–6 are commented out in the source) | `Outputs_predict_batch_files/` | required |

Both go through `Claude_predict_folder.py` (imported, not an entry point): region `ap-southeast-1`,
model `global.anthropic.claude-sonnet-4-6`.

**Hazard attribute extraction & labelling** — prompt (LLM branch only):
`Assay_attr_Extraction_Codes/hazard_no_hazards_prompt_V3.txt`

Records that carry a structured `hazards` field are always labelled by table lookup. Records that
only carry a subject need the no-hazards branch, which is available as n-gram keyword rules or as
an LLM. The four scripts differ only in that branch:

| Script | `hazards` field present | subject only | Bedrock token |
|---|---|---|---|
| `Assay_attr_Extraction_Codes/hazard_eval_gt2024.py` — test on 2024 ground truth | table lookup | n-gram keyword rules | not needed |
| `Assay_attr_Extraction_Codes/hazard_eval.py` — predict on one 2024 batch file (`INPUT_FILE` to change) | table lookup | n-gram keyword rules | not needed |
| `Assay_attr_Extraction_Codes/hazard_eval_gt2024_spec.py` — test on 2024 ground truth | table lookup | Claude on Bedrock | required |
| `Assay_attr_Extraction_Codes/hazard_predict_folder_llm.py` — predict hazard on a folder | table lookup | Claude on Bedrock | required |

- The two keyword-rule scripts read `hazard/no_hazards_mapping1.json` and
  `hazard/no_hazards_mapping2.json`; `hazard_eval_gt2024.py` imports `hazard_eval.py` and reuses
  its rule engine. Output dirs: `Outputs_hazard_eval_gt2024/` and
  `Outputs_hazard_predict_rule_only_v6/`.
- The two LLM scripts pick the label from the 1021 allowed labels, and never call the LLM on the
  `hazards` branch. They accept `AWS_BEARER_TOKEN_BEDROCK` (Bedrock; default region `us-east-1`,
  override with `AWS_REGION`) **or** `ANTHROPIC_API_KEY` (Anthropic direct).
  `hazard_eval_gt2024_spec.py` writes `Outputs_hazard_eval_gt2024_spec/`;
  `hazard_predict_folder_llm.py` reads `INPUT_DIR` — a folder of product-label outputs, **not
  included in this repo** — and writes `<folder-name>_hazard_llm/` next to it.

The table lookup resolves `hazard_label` from `hazard/has_hazards_mapping_hazard_label.json`
(1024 keys; an empty value in that file means the label is the key itself); `hazard_category_label`
is always derived from `hazard_label` through
`hazard/mapping_hazard_label_to_hazard_category_label.json` (1021 keys — these keys are also the
allowed label set given to the LLM).

**Development tools** (not wired into Docker): `hazard_model_bakeoff.py` compares Bedrock models on
a fixed sample of the ground-truth no-hazards records; `bedrock_probe_nonanthropic.py` probes which
non-Anthropic models the current credentials can reach.

## Run with Docker (recommended)

Everything (code, prompts, data) is bundled, so it installs and runs anywhere — no manual
dependency setup. See **[`README_docker.md`](README_docker.md)** for full details.

```bash
docker build -t rasff_labels .

docker run --rm rasff_labels                          # 2024 product-label test (default)
docker run --rm rasff_labels test-product --limit 20  # quick product test
docker run --rm rasff_labels test-hazard              # 2024 hazard test (keyword rules, no token)
docker run --rm rasff_labels predict-product          # product predict on a 2024 batch file
docker run --rm rasff_labels predict-hazard           # hazard predict (table lookup + keyword rules)
docker run --rm rasff_labels predict-hazard-llm       # hazard predict, no-hazards branch via LLM
```

Bedrock credentials are **not** committed. Pass your token at run time:

```bash
docker run --rm -e AWS_BEARER_TOKEN_BEDROCK="<your-token>" rasff_labels
```

`test-hazard` and `predict-hazard` use table lookup + keyword rules only and need no token.

## Run locally (without Docker)

The scripts auto-detect the repo root, so from the repo directory:

```bash
pip install -r Assay_attr_Extraction_Codes/requirements.txt
export AWS_BEARER_TOKEN_BEDROCK="<your-token>"   # not needed for hazard_eval.py / hazard_eval_gt2024.py

python Assay_attr_Extraction_Codes/evaluate_product_label.py --limit 20
python Assay_attr_Extraction_Codes/hazard_eval.py
```

## Layout

```
.
├── Assay_attr_Extraction_Codes/   # 5 Docker entry scripts (evaluate_product_label, Claude_eval,
│                                  #   hazard_eval, hazard_eval_gt2024, hazard_predict_folder_llm)
│                                  # + hazard_eval_gt2024_spec, hazard_model_bakeoff,
│                                  #   bedrock_probe_nonanthropic (run directly, not via Docker)
│                                  # + Claude_predict_folder.py (imported, not an entry point)
│                                  # + prompts, hazard_canon_config.json, entrypoint.sh, requirements
├── FIND-food-recall-data-main_V2/ # product_labels.txt + rasff_data_2024_batch1..6.json
├── hazard/                        # hazard gold + mapping JSON
├── rasff_2024_ground_truth_labels.json   # 2024 ground truth (read by the four test scripts)
├── Dockerfile, .dockerignore, .gitignore
└── README.md, README_docker.md
```

All paths derive from `RASFF_ROOT` (the repo root locally; `/app` inside the Docker image).
