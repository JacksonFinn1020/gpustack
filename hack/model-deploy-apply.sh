#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

gpustack::log::info "model-deploy apply"
uv run python "${ROOT_DIR}/contrib/model-deploy/apply.py" "$@"
