# Running in Docker

This repo is packaged so the product-label and hazard-label/category pipeline installs
and runs cleanly anywhere — no manual dependency setup. Code, prompts and data are all
included.

## What runs
Modes are dispatched by `Assay_attr_Extraction_Codes/entrypoint.sh`; extra arguments are passed
straight through to the Python script.

| Mode | Script | Purpose | Output dir | LLM? |
|------|--------|---------|------------|------|
| `test-product` (default) | `evaluate_product_label.py` | product_label vs 2024 ground truth (Precision/Recall/F1) | `eval_product_label_output/` | ✅ Bedrock |
| `test-hazard` | `hazard_eval_gt2024.py` | hazard label/category vs 2024 ground truth | `Outputs_hazard_eval_gt2024/` | ❌ table lookup + keyword rules |
| `predict-product` | `Claude_eval.py` | predict product_label on a 2024 batch file (batch1 by default; batch2–6 are commented out in the source) | `Outputs_predict_batch_files/` | ✅ Bedrock |
| `predict-hazard` | `hazard_eval.py` | predict hazard label/category on one 2024 batch file (batch1 by default, `-e INPUT_FILE=...` to change) | `Outputs_hazard_predict_rule_only_v6/` | ❌ table lookup + keyword rules |
| `predict-hazard-llm` | `hazard_predict_folder_llm.py` | same, but records without a structured `hazards` field go to the LLM instead of the keyword rules. Reads a *folder* of product-label outputs, which is not in this repo — set `INPUT_DIR` | `<folder-name>_hazard_llm/` | ✅ Bedrock (no-hazards branch only) |
| `shell` | — | drop into bash for debugging | — | — |

Records that carry a structured `hazards` field are always resolved by table lookup, in every
mode. The modes differ only in how records with just a subject are handled: n-gram keyword rules
over `hazard/no_hazards_mapping1.json` + `no_hazards_mapping2.json` (`hazard_eval.py`,
`hazard_eval_gt2024.py`) or Claude (`hazard_predict_folder_llm.py`).

`hazard_eval_gt2024_spec.py` (ground-truth test with the LLM no-hazards branch),
`hazard_model_bakeoff.py` and `bedrock_probe_nonanthropic.py` have no Docker mode — run them with
`docker run --rm rasff_labels shell` or directly with `python`.

## Hazard accuracy on the 2024 ground truth
Measured on the `rasff_2024_ground_truth_labels.json` in this image: 5249 records — 3892 with a
structured `hazards` field, 1357 subject-only. `exact` = full set equality on the record.

| Branch | field | has_hazards | no_hazards | overall |
|---|---|---:|---:|---:|
| LLM (`hazard_eval_gt2024_spec.py`, prompt V3) | `hazard_label` exact | 0.9990 | **0.8968** | **0.9726** |
| LLM (`hazard_eval_gt2024_spec.py`, prompt V3) | `hazard_category_label` exact | 0.9985 | **0.9256** | **0.9796** |
| keyword rules (`test-hazard`) | `hazard_label` exact | 0.9928 | 0.5099 | 0.8680 |
| keyword rules (`test-hazard`) | `hazard_category_label` exact | 0.9949 | 0.7170 | 0.9230 |

Full per-segment Jaccard and micro P/R/F1, plus the caveats that matter when comparing the two
branches, are in [`README.md`](README.md#hazard-accuracy-on-the-2024-ground-truth). In short: the
`has_hazards` segment never calls an LLM in either branch (it is mapping-file coverage), and the
keyword branch's inputs overlap the 2024 test set, so its numbers are optimistic and not directly
comparable. Reproduce with:

```bash
docker run --rm rasff_labels test-hazard                      # keyword rules, no token
docker run --rm -e AWS_BEARER_TOKEN_BEDROCK="<token>" \
  -e CLAUDE_MODEL=us.amazon.nova-pro-v1:0 rasff_labels \
  shell -c "python hazard_eval_gt2024_spec.py"                # LLM branch
```

The LLM figures were produced with `us.amazon.nova-pro-v1:0` rather than the
`global.anthropic.claude-sonnet-4-6` default, because Anthropic model access on the AWS account used
for the run is still gated.

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
docker run --rm rasff_labels test-hazard              # 2024 hazard test (no token needed)
docker run --rm rasff_labels predict-product          # product predict on a 2024 batch file
docker run --rm rasff_labels predict-hazard           # hazard predict (no token needed)
docker run --rm rasff_labels predict-hazard-llm       # hazard predict, no-hazards branch via LLM
```

`predict-hazard-llm` reads a folder of product-label outputs (`-e INPUT_DIR=...`) and writes
`<folder-name>_hazard_llm/` next to it (a trailing `_product` in the folder name is replaced).

### Bedrock credentials
No token is committed:
- `evaluate_product_label.py` and `Claude_eval.py` raise
  `ValueError: Please set a valid AWS_BEARER_TOKEN_BEDROCK.` when the variable is unset (the
  literal placeholder `REPLACE_WITH_YOUR_AWS_BEARER_TOKEN_BEDROCK` in `Claude_eval.py` /
  `Claude_predict_folder.py` is only a fallback string, never a working credential).
- `hazard_predict_folder_llm.py` raises `ValueError: Set ANTHROPIC_API_KEY (Anthropic direct) or
  AWS_BEARER_TOKEN_BEDROCK (Bedrock).`

```bash
docker run --rm -e AWS_BEARER_TOKEN_BEDROCK="<your-token>" rasff_labels
```

Model `global.anthropic.claude-sonnet-4-6` throughout. Region differs by script: the product
scripts pin `ap-southeast-1`; `hazard_predict_folder_llm.py` defaults to `us-east-1` and honours
`-e AWS_REGION=...`. It also accepts `-e ANTHROPIC_API_KEY=...` to call the Anthropic API directly
instead of Bedrock. `test-hazard` and `predict-hazard` need no token at all.

### Cheap smoke testing
```bash
docker run --rm rasff_labels test-product --limit 5      # 5 LLM calls
docker run --rm -e DEBUG_N=5 rasff_labels predict-product
```

### Getting results out
Outputs are written under `Assay_attr_Extraction_Codes/` — see the output-dir column above. Mount
a volume to keep them:
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
├── Assay_attr_Extraction_Codes/   # the 5 entry scripts + hazard_eval_gt2024_spec,
│                                  #   hazard_model_bakeoff, bedrock_probe_nonanthropic (no Docker
│                                  #   mode) + Claude_predict_folder.py (imported, not an entry
│                                  #   point) + prompts, hazard_canon_config.json, entrypoint.sh,
│                                  #   requirements.txt
├── FIND-food-recall-data-main_V2/ # product_labels.txt + rasff_data_2024_batch1..6.json
├── hazard/                        # hazard gold + mapping json
├── rasff_2024_ground_truth_labels.json  # 2024 ground truth (read by the four test scripts)
├── Dockerfile, .dockerignore, .gitignore
└── README.md, README_docker.md
```
All paths derive from `RASFF_ROOT` (the repo root locally; `/app` in the image).
