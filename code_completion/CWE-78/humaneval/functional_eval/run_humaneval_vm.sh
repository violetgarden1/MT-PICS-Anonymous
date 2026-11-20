#!/usr/bin/env bash
# Run only inside the dedicated VMware Ubuntu guest, with Docker isolation.
set -euo pipefail

ROOT="${HUMANEVAL_SANDBOX_ROOT:-${HOME}/humaneval-sandbox}"
IMAGE="ganler/evalplus:v0.3.1"
EXPECTED_IMAGE="sha256:57f7b4f28373c938f23ed51c412419974fc1ba7570979f19cfee3678fe9a64c8"
DATA="${ROOT}/data/HumanEvalPlus-v0.1.10.jsonl.gz"
EXPECTED_DATA="272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
PATCH="${ROOT}/evalplus_patch/__init__.py"
EXPECTED_PATCH="905bef8d559bfd955780b670c93a76f5b7a4a09b2a521170ff67de15d389ff96"
ARM="${1:-}"
case "$ARM" in
  smoke) SOURCE="${ROOT}/smoke/official_reference_smoke.jsonl" ;;
  base|pws|composite) SOURCE="${ROOT}/inputs/${ARM}.jsonl" ;;
  *) echo "Usage: $0 {smoke|base|pws|composite}" >&2; exit 2 ;;
esac

[[ -f "$DATA" && -f "$SOURCE" && -f "$PATCH" ]] || { echo "Missing dataset or input samples" >&2; exit 1; }
[[ "$(sha256sum "$DATA" | cut -d' ' -f1)" == "$EXPECTED_DATA" ]] || { echo "Dataset hash mismatch" >&2; exit 1; }
[[ "$(sha256sum "$PATCH" | cut -d' ' -f1)" == "$EXPECTED_PATCH" ]] || { echo "EvalPlus patch hash mismatch" >&2; exit 1; }
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE")" == "$EXPECTED_IMAGE" ]] || { echo "Image ID mismatch" >&2; exit 1; }
if [[ "$ARM" == smoke ]]; then
  [[ "$(sha256sum "$SOURCE" | cut -d' ' -f1)" == "b7e7e19afa6f1668ab69029a172c35512a822438b4709dd1243ca452de33bb21" ]] || { echo "Smoke input hash mismatch" >&2; exit 1; }
else
  python3 - "$SOURCE" "${ROOT}/inputs/conversion_manifest.json" "$ARM" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); manifest=json.loads(pathlib.Path(sys.argv[2]).read_text(encoding='utf-8'))
expected=manifest['arms'][sys.argv[3]]['samples_sha256']
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, 'Input hash mismatch'
rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(rows)==164 and len({row['task_id'] for row in rows})==164, 'Expected 164 distinct tasks'
PY
fi

mkdir -p "${ROOT}/runs"
WORK="$(mktemp -d "${ROOT}/runs/${ARM}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
cp -- "$SOURCE" "$WORK/samples.jsonl"
printf 'group=%s\nimage=%s\nimage_id=%s\ndataset_sha256=%s\nsource_sha256=%s\npatch_sha256=%s\n' \
  "$ARM" "$IMAGE" "$EXPECTED_IMAGE" "$EXPECTED_DATA" "$(sha256sum "$SOURCE" | cut -d' ' -f1)" "$EXPECTED_PATCH" > "$WORK/run_metadata.txt"
echo "Evaluation output: $WORK"
docker run --rm --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --pids-limit=128 --memory=2200m --cpus=2 --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=768m \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e PYTHONDONTWRITEBYTECODE=1 \
  -e HUMANEVAL_OVERRIDE_PATH=/benchmark/HumanEvalPlus-v0.1.10.jsonl.gz \
  --mount "type=bind,source=${ROOT}/data,target=/benchmark,readonly" \
  --mount "type=bind,source=${PATCH},target=/usr/local/lib/python3.11/site-packages/evalplus/eval/__init__.py,readonly" \
  --mount "type=bind,source=${WORK},target=/work" --workdir /work \
  "$IMAGE" evalplus.evaluate --dataset humaneval --samples /work/samples.jsonl --base-only --parallel 1 \
  2>&1 | tee "$WORK/evaluation.log"
echo "Completed: $WORK"
