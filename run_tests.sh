#!/usr/bin/env bash
#
# Test runner for Reachy Personal Assistant
#
# Usage:
#   ./run_tests.sh              # Run unit tests only (fast, no services needed)
#   ./run_tests.sh unit         # Same as above
#   ./run_tests.sh integration  # Run integration tests (needs vLLM running)
#   ./run_tests.sh all          # Run everything
#   ./run_tests.sh all -x       # Run everything, stop on first failure
#
# Extra pytest arguments are passed through:
#   ./run_tests.sh unit -k "test_config"     # Run only config tests
#   ./run_tests.sh unit -s                   # Show stdout/print output
#   ./run_tests.sh all --tb=long             # Verbose tracebacks
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUITE="${1:-unit}"
shift 2>/dev/null || true  # Remaining args passed to pytest

case "$SUITE" in
    unit)
        echo "=== Running UNIT tests (no external services needed) ==="
        python3 -m pytest tests/unit/ -m "not integration" "$@"
        ;;
    integration)
        echo "=== Running INTEGRATION tests (vLLM must be running) ==="
        python3 -m pytest tests/integration/ -m "integration" "$@"
        ;;
    all)
        echo "=== Running ALL tests ==="
        echo ""
        echo "--- Unit tests ---"
        python3 -m pytest tests/unit/ -m "not integration" "$@"
        echo ""
        echo "--- Integration tests ---"
        python3 -m pytest tests/integration/ "$@"
        ;;
    *)
        echo "Usage: $0 {unit|integration|all} [pytest args...]"
        exit 1
        ;;
esac
