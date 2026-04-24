#!/usr/bin/env bash
# Deploy the OpenEnv to a Hugging Face Space.
#
# Prerequisites:
#   - HF account with write token: https://huggingface.co/settings/tokens
#   - `pip install huggingface_hub`
#   - HF_USERNAME set (your org/username)
#   - HF_TOKEN set (your write token) — NEVER commit this file with the token
#
# Usage:
#   export HF_USERNAME=<your-username>
#   export HF_TOKEN=hf_xxx
#   bash scripts/deploy_to_hf.sh
#
# On first run:
#   1. Creates the Space at https://huggingface.co/spaces/$HF_USERNAME/portfolio-env
#      with SDK=docker (so our Dockerfile builds & serves)
#   2. Clones the Space repo to /tmp/portfolio-env-space
#   3. Copies our repo contents in (excluding heavy dirs)
#   4. Commits + pushes → HF auto-builds the image and starts the Space
#
# On re-run: updates the Space with latest local state.

set -euo pipefail

: "${HF_USERNAME:?Set HF_USERNAME (e.g. export HF_USERNAME=myhandle)}"
: "${HF_TOKEN:?Set HF_TOKEN (write token from https://huggingface.co/settings/tokens)}"

SPACE_NAME="portfolio-env"
REPO_ID="$HF_USERNAME/$SPACE_NAME"
WORK_DIR="/tmp/$SPACE_NAME-space"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "──────────────────────────────────────────────────────────────────"
echo "  Deploying $SRC_DIR → https://huggingface.co/spaces/$REPO_ID"
echo "──────────────────────────────────────────────────────────────────"

# 1. Ensure Space exists (idempotent — creates if missing)
python - <<PYEOF
import os
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ['HF_TOKEN'])
try:
    api.repo_info(repo_id="$REPO_ID", repo_type="space")
    print("  Space already exists: $REPO_ID")
except Exception:
    create_repo(
        repo_id="$REPO_ID",
        repo_type="space",
        space_sdk="docker",
        token=os.environ['HF_TOKEN'],
        exist_ok=True,
    )
    print("  Created Space: $REPO_ID (sdk=docker)")
PYEOF

# 2. Clone or refresh the Space repo working dir
if [ -d "$WORK_DIR/.git" ]; then
    echo "  Refreshing existing clone at $WORK_DIR"
    (cd "$WORK_DIR" && git fetch && git reset --hard origin/main 2>/dev/null || git reset --hard origin/master 2>/dev/null || true)
else
    rm -rf "$WORK_DIR"
    git clone "https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$REPO_ID" "$WORK_DIR"
fi

# 3. Copy (rsync) the deployable files into the Space repo
#    Exclude heavy / irrelevant dirs.
rsync -av --delete \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.egg-info' \
    --exclude='round_1' \
    --exclude='.playwright-mcp' \
    --exclude='sft_traces/_*.log' \
    --exclude='notebooks/*.ipynb_checkpoints' \
    "$SRC_DIR/" "$WORK_DIR/"

# 4. Ensure README has HF Space metadata frontmatter (idempotent prepend)
if ! head -1 "$WORK_DIR/README.md" | grep -q '^---'; then
    TMP=$(mktemp)
    cat > "$TMP" <<'FRONTMATTER'
---
title: Reasoning-Under-Constraints OpenEnv
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - rl
  - grpo
  - qwen3
  - portfolio-reasoning
---

FRONTMATTER
    cat "$WORK_DIR/README.md" >> "$TMP"
    mv "$TMP" "$WORK_DIR/README.md"
    echo "  Added HF Space YAML frontmatter to README.md"
fi

# 5. Commit + push → HF auto-builds
cd "$WORK_DIR"
git config user.email "$HF_USERNAME@users.noreply.huggingface.co"
git config user.name "$HF_USERNAME"
git add -A
if git diff --cached --quiet; then
    echo "  No changes to push."
else
    git commit -m "deploy: $(date -u +%Y-%m-%dT%H:%M:%SZ) snapshot from $(git -C "$SRC_DIR" rev-parse --short HEAD)"
    git push
fi

echo ""
echo "──────────────────────────────────────────────────────────────────"
echo "  Deployed. Check build logs + live Space at:"
echo "    https://huggingface.co/spaces/$REPO_ID"
echo "  First build takes 3-8 min. After build:"
echo "    /health     → liveness"
echo "    /metadata   → env description"
echo "    /ws         → OpenEnv WebSocket protocol"
echo "    /docs       → interactive API"
echo "──────────────────────────────────────────────────────────────────"
