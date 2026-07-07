#!/bin/bash
# Dispatcher for the RASFF product-label / hazard pipeline inside the container.
#
# Modes (first arg; remaining args are passed through to the Python script):
#   test-product     evaluate product_label on the 2024 ground truth  (P/R/F1)   [DEFAULT]
#   test-hazard      evaluate hazard_label/category on the 2024 ground truth
#   predict-product  predict product_label on the 2024 batch file(s)
#   predict-hazard   predict hazard_label/category on the 2024 batch file(s)
#   shell            drop into bash (debugging)
#
# Examples:
#   docker run --rm rasff_labels                              # 2024 product-label test (full)
#   docker run --rm rasff_labels test-product --limit 20      # quick product test
#   docker run --rm rasff_labels test-hazard                  # 2024 hazard test
#   docker run --rm rasff_labels predict-hazard               # hazard predict (table-lookup, no token needed)
set -euo pipefail
cd "${RASFF_ROOT:-/app}/Assay_attr_Extraction_Codes"

MODE="${1:-test-product}"
shift || true

case "$MODE" in
  test-product)    exec python evaluate_product_label.py "$@" ;;
  test-hazard)     exec python hazard_eval_gt2024.py "$@" ;;
  predict-product) exec python Claude_eval.py "$@" ;;
  predict-hazard)  exec python hazard_eval.py "$@" ;;
  shell)           exec bash "$@" ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Valid modes: test-product | test-hazard | predict-product | predict-hazard | shell" >&2
    exit 2
    ;;
esac
