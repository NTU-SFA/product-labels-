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

### Hazard accuracy on the 2024 ground truth

Measured on `rasff_2024_ground_truth_labels.json` as shipped here: 5249 records —
3892 with a structured `hazards` field, 1357 subject-only. `exact` is full set equality on the
record (the primary metric); P/R/F1 are micro-averaged over labels.

**LLM branch** — `hazard_eval_gt2024_spec.py`, prompt `hazard_no_hazards_prompt_V3.txt`,
1021-label space:

| field | segment | n | exact | jaccard | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `hazard_label` | has_hazards | 3892 | **0.9990** | 0.9993 | 0.9996 | 0.9987 | 0.9991 |
| `hazard_label` | no_hazards | 1357 | **0.8968** | 0.9138 | 0.9195 | 0.9275 | 0.9235 |
| `hazard_label` | overall | 5249 | **0.9726** | 0.9772 | 0.9827 | 0.9837 | 0.9832 |
| `hazard_category_label` | has_hazards | 3892 | **0.9985** | 0.9988 | 0.9998 | 0.9985 | 0.9991 |
| `hazard_category_label` | no_hazards | 1357 | **0.9256** | 0.9382 | 0.9495 | 0.9481 | 0.9488 |
| `hazard_category_label` | overall | 5249 | **0.9796** | 0.9831 | 0.9874 | 0.9862 | 0.9868 |

147 records mismatch in total. Notes on reading these numbers:

- The `has_hazards` segment involves **no LLM** — it is the mapping file's coverage, so 0.9990 is a
  property of `hazard/has_hazards_mapping_hazard_label.json`, not of the model. The remaining gap on
  category is two labels with no entry in `mapping_hazard_label_to_hazard_category_label.json`:
  `Bacterial contamination` and `Toxin unknown`.
- Only the `no_hazards` segment measures the LLM. It was run with **`us.amazon.nova-pro-v1:0`**, not
  the `global.anthropic.claude-sonnet-4-6` default, because Anthropic model access on the AWS
  account used for the run is still gated. Set `CLAUDE_MODEL` to reproduce with another model;
  0 LLM errors in this run.

**Keyword-rule branch** — `hazard_eval_gt2024.py` (`test-hazard`), same ground truth, no LLM:

| field | has_hazards | no_hazards | overall |
|---|---:|---:|---:|
| `hazard_label` exact | 0.9928 | 0.5099 | 0.8680 |
| `hazard_category_label` exact | 0.9949 | 0.7170 | 0.9230 |

Two caveats make this branch **not** a fair comparison:

1. **The test set leaks into its inputs.** It builds its allowed-label sets and preferred lookups
   from `hazard/rasff_data_2020_to_2026_*_with_labels.json`, which already contain the 2024 records
   being scored — 3892/3892 (100%) of the `has_hazards` records and 1038/1357 (76.5%) of the
   `no_hazards` ones. Its `has_hazards` figure is therefore optimistic.
2. **Its rules target the previous taxonomy.** They were tuned against the older 936-label space, so
   they under-perform on the current 1021-label ground truth. This is the legacy path; the LLM
   branch above is the current one.

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
