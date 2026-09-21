#!/usr/bin/env bash
# Export a trained tracker checkpoint and point booster_deploy at it.
#
#   scripts/sync_latest_checkpoint.sh                 # newest checkpoint of any run
#   scripts/sync_latest_checkpoint.sh <run_dir>       # newest checkpoint of that run
#   scripts/sync_latest_checkpoint.sh <run_dir> 9000  # that checkpoint
#
# Exports with play.py (which writes exported/<run>.pt), keeps a copy named after the iteration, copies it into
# booster_deploy/tasks/beyond_mimic/models/, and updates that clip's LARGEBOX_TRACKER_OVERFIT entry in place.
set -euo pipefail

TRAIN_DIR="$HOME/booster_train"
DEPLOY_DIR="$HOME/booster_deploy/tasks/beyond_mimic"
cd "$TRAIN_DIR"

RUN="${1:-}"
if [[ -z "$RUN" ]]; then
    RUN="$(dirname "$(ls -t logs/rsl_rl/*/*/model_*.pt | head -1)")"
fi
RUN="${RUN%/}"
ITER="${2:-}"
if [[ -z "$ITER" ]]; then
    ITER="$(basename "$(ls "$RUN"/model_*.pt | sort -V | tail -1)" .pt)"
    ITER="${ITER#model_}"
fi
CKPT="$RUN/model_$ITER.pt"
[[ -f "$CKPT" ]] || { echo "no such checkpoint: $CKPT" >&2; exit 1; }

EXPERIMENT="$(basename "$(dirname "$RUN")")"   # e.g. k1_largebox_tracker_sub15_suitcase_017_stand
RUN_NAME="$(basename "$RUN")"                  # e.g. 2026-09-19_16-52-45_overfit_suitcase
CLIP="${EXPERIMENT#k1_largebox_tracker}"
CLIP="${CLIP#_}"                               # "" for the multi-clip task, else sub15_suitcase_017_stand
if [[ -n "$CLIP" ]]; then
    TASK="Booster-K1-Largebox-Tracker-$CLIP-v0-Play"
else
    TASK="Booster-K1-Largebox-Tracker-v0-Play"
fi

echo "[SYNC] run   : $RUN"
echo "[SYNC] ckpt  : model_$ITER.pt"
echo "[SYNC] task  : $TASK"

~/env_isaaclab/bin/python scripts/rsl_rl/play.py --task "$TASK" --headless --num_envs 1 \
    --checkpoint "$TRAIN_DIR/$CKPT" 2>&1 | grep -E "Loading model checkpoint|Error" || true

EXPORTED="$RUN/exported/${EXPERIMENT}_${RUN_NAME}"
[[ -f "$EXPORTED.pt" ]] || { echo "play.py did not write $EXPORTED.pt" >&2; exit 1; }
cp "$EXPORTED.pt" "$RUN/exported/model_$ITER.pt"
[[ -f "$EXPORTED.onnx" ]] && cp "$EXPORTED.onnx" "$RUN/exported/model_$ITER.onnx"

DEPLOY_NAME="${EXPERIMENT}_${RUN_NAME}_model_${ITER}.pt"
cp "$RUN/exported/model_$ITER.pt" "$DEPLOY_DIR/models/$DEPLOY_NAME"
echo "[SYNC] model : $DEPLOY_DIR/models/$DEPLOY_NAME"

# Point this clip's overfit entry at the new model. The key is the motion file stem whose task name is $CLIP,
# matching the discovery rule in booster_deploy/tasks/beyond_mimic/__init__.py.
CLIP="$CLIP" DEPLOY_NAME="$DEPLOY_NAME" DEPLOY_DIR="$DEPLOY_DIR" TRAIN_DIR="$TRAIN_DIR" python3 - <<'PY'
import glob, os, re

clip, model, deploy_dir, train_dir = (os.environ[k] for k in ("CLIP", "DEPLOY_NAME", "DEPLOY_DIR", "TRAIN_DIR"))
init = os.path.join(deploy_dir, "__init__.py")
src = open(init).read()

if not clip:
    src = re.sub(r'(LARGEBOX_TRACKER_CHECKPOINT = ")[^"]*(")', rf'\g<1>models/{model}\g<2>', src, count=1)
    print(f"[SYNC] cfg   : LARGEBOX_TRACKER_CHECKPOINT -> models/{model}")
else:
    folders = [
        os.path.join(deploy_dir, "motions", "largebox_tracker"),
        os.path.join(train_dir, "booster_assets", "motions", "K1", "tracker", "npz"),
    ]
    stems = [
        os.path.splitext(os.path.basename(p))[0]
        for folder in folders
        for p in sorted(glob.glob(os.path.join(folder, "*.npz")))
    ]
    match = [s for s in stems if s.replace("_largebox", "").replace("_hold", "") == clip]
    if not match:
        raise SystemExit(f"[SYNC] no motion file maps to task name {clip!r}; checked {folders}")
    key = match[0]
    entry = f'    "{key}": "models/{model}",\n'
    pattern = re.compile(rf'^    "{re.escape(key)}": "[^"]*",\n', re.M)
    if pattern.search(src):
        src = pattern.sub(entry, src, count=1)
    else:
        src = src.replace("LARGEBOX_TRACKER_OVERFIT = {\n", "LARGEBOX_TRACKER_OVERFIT = {\n" + entry, 1)
    print(f"[SYNC] cfg   : LARGEBOX_TRACKER_OVERFIT[{key!r}] -> models/{model}")
    print(f"[SYNC] task  : k1_largebox_{clip}_overfit")

open(init, "w").write(src)
PY
