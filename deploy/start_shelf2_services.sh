#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Select the shelf2-only data without changing any production configuration.
export RUNTIME_CONFIG_FILE="$PROJECT/agent/config/runtime.shelf2.yaml"
export SKU_CATALOG_PATH="$PROJECT/perception/sku/products.shelf2.json"
export INITIAL_SCAN_ROOT="$PROJECT/agent/output/task0-shelf2"
export PRODUCT_HAND_OPTIONS_PATH="$PROJECT/agent/config/product-hand-options.shelf2.yaml"
export INSPECT_SKU_CATALOG_PATH="$SKU_CATALOG_PATH"

exec "$SCRIPT_DIR/start_all_services.sh" "$@"
