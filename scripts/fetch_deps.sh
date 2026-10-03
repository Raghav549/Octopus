#!/usr/bin/env bash
# Fetch and pin the optional third-party dependency (llama.cpp) for Octopus.
#
# Octopus builds and tests fully without this script; the dependency is only
# needed for (a) real SLM inference (OCT_WITH_LLAMA=ON) and (b) the vocabulary /
# reference fixtures used by the tokenizer tests and benchmarks.
#
# The revision is pinned; nothing here is a floating branch.
# SPDX-License-Identifier: MIT
set -euo pipefail

REPO="https://github.com/ggml-org/llama.cpp"
TAG="b11371"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/llama.cpp"

if [[ -d "${DEST}/.git" ]]; then
  echo "llama.cpp already present at ${DEST}"
  echo "  HEAD: $(git -C "${DEST}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "  tag:  $(git -C "${DEST}" describe --tags 2>/dev/null || echo unknown)"
  exit 0
fi

echo "cloning ${REPO} at tag ${TAG} -> ${DEST}"
mkdir -p "$(dirname "${DEST}")"
git clone --depth 1 --branch "${TAG}" "${REPO}" "${DEST}"

echo
echo "done. to link a real inference backend, build it and configure Octopus with:"
echo "  cmake -B build -S . -DOCT_WITH_LLAMA=ON -DOCT_LLAMA_PREBUILT_DIR=${DEST}/build"
echo "without that, Octopus reports the backend as unavailable and refuses to"
echo "answer without an explicit --allow-stub (see docs/LIMITATIONS.md)."
