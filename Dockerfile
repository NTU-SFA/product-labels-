# RASFF product-label + hazard-label/category pipeline.
# Self-contained: code + data + prompts are all baked into the image.
#
# Build (from the repo root):
#   docker build -t rasff_labels .
# Run the 2024 product-label test (P/R/F1 on the ground truth):
#   docker run --rm rasff_labels
# Quick smoke test on 20 records:
#   docker run --rm rasff_labels test-product --limit 20
# 2024 hazard test:
#   docker run --rm rasff_labels test-hazard
# Predict on the 2024 batch file:
#   docker run --rm rasff_labels predict-product      # product_label (LLM)
#   docker run --rm rasff_labels predict-hazard       # hazard label/category (rule-only, no token)
# Override the Bedrock token (recommended over the baked-in default):
#   docker run --rm -e AWS_BEARER_TOKEN_BEDROCK=... rasff_labels

FROM python:3.10-slim

# RASFF_ROOT tells the code where the project root is; every path derives from it.
ENV RASFF_ROOT=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Python dependencies (own layer for build caching) ---
COPY Assay_attr_Extraction_Codes/requirements.txt /app/Assay_attr_Extraction_Codes/requirements.txt
RUN pip install --no-cache-dir -r /app/Assay_attr_Extraction_Codes/requirements.txt

# --- Everything else (code + prompts + data). .dockerignore keeps outputs/caches out. ---
COPY . /app
RUN chmod +x /app/Assay_attr_Extraction_Codes/entrypoint.sh

WORKDIR /app/Assay_attr_Extraction_Codes
ENTRYPOINT ["/app/Assay_attr_Extraction_Codes/entrypoint.sh"]
CMD ["test-product"]
