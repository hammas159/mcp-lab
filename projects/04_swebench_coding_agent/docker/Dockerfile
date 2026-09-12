# Minimal sandbox for running one SWE-bench-Lite instance's own test suite
# against a model-generated patch. Deliberately generic (no repo-specific
# system deps) because the two instances we evaluate (flask, requests) are
# pure-Python with no compiled extensions.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /workspace
