#!/usr/bin/env bash
# Start the API. With a checkpoint:  TRIAGE_CHECKPOINT=runs/synthetic/best.pt ./scripts/serve.sh
# Without one, the stub model (random probabilities) is used.
cd "$(dirname "$0")/.." || exit 1
exec uvicorn "triage.serving.app:create_app" --factory --host 0.0.0.0 --port 8000
