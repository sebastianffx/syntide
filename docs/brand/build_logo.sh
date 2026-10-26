#!/usr/bin/env bash
# Rebuild the transparent lockups and editable repository cover in PhotoCraft.
# Inputs: syntide-mark-source.png (generated artwork), syntide-convergence.png.
# Set PHOTOCRAFT_CLI if PhotoCraft is installed elsewhere.
set -euo pipefail

asset_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pc="${PHOTOCRAFT_CLI:-/Applications/photocraft-cli-0.3.0-macos-universal/photocraft-cli}"
scratch="$(mktemp -d "${SYNTIDE_BRAND_TMP:-${TMPDIR:-/tmp}}/syntide-logo.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
cd "$asset_dir"

# Remove a faint colored fringe at alpha edges; retain the original source.
"$pc" run syntide-mark-source.png \
  --cmd layer.matting.defringe --params '{"width":3}' \
  --out syntide-mark.png

# Crop transparent padding for the lockups; keep the square mark for avatars.
"$pc" run syntide-mark.png \
  --cmd image.crop \
  --params '{"x":118,"y":365,"width":1020,"height":540}' \
  --out "$scratch/mark-crop.png"

# JSON encoding keeps paths containing spaces or quotation marks safe.
logo_placement="$(python3 -c 'import json,sys; print(json.dumps({"path":sys.argv[1],"scale":38,"center":[270,150]}))' "$scratch/mark-crop.png")"
cover_placement="$(python3 -c 'import json,sys; print(json.dumps({"path":sys.argv[1],"scale":22.8,"center":[294,145]}))' "$scratch/mark-crop.png")"

"$pc" run \
  --new '{"width":1200,"height":300,"background":"transparent","resolution":72,"name":"Syntide logo"}' \
  --cmd file.placeEmbedded --params "$logo_placement" \
  --cmd layer.renameLayer --params '{"name":"Syntide wave mark"}' \
  --cmd type.create \
  --params '{"x":520,"y":202,"text":"syntide","name":"Syntide wordmark","font":"Inter","weight":600,"size":180,"color":"#192e3b","tracking":-40}' \
  --cmd file.saveACopy --params '{"path":"syntide-logo.png"}' \
  --cmd file.saveACopy --params '{"path":"syntide-logo.psd","layers":true}' \
  --out syntide-logo.pcraft

# The active layer is live type in the saved master.
"$pc" run syntide-logo.pcraft \
  --cmd type.setStyle --params '{"color":"#ffffff"}' \
  --cmd file.saveACopy --params '{"path":"syntide-logo-light.png"}' \
  --out syntide-logo-light.pcraft

# Preserve the Blender render as the unmodified background layer.
"$pc" run syntide-convergence.png \
  --cmd layer.renameLayer --params '{"name":"Convergence — Blender render"}' \
  --cmd file.placeEmbedded --params "$cover_placement" \
  --cmd layer.renameLayer --params '{"name":"Syntide wave mark"}' \
  --cmd type.create \
  --params '{"x":444,"y":176,"text":"syntide","name":"Syntide wordmark","font":"Inter","weight":600,"size":108,"color":"#ffffff","tracking":-40}' \
  --cmd type.create \
  --params '{"x":180,"y":255,"text":"Synthetic time-series generation","name":"Descriptor","font":"Inter","weight":400,"size":30,"color":"#a8bcc4","tracking":5}' \
  --cmd file.saveACopy --params '{"path":"syntide-cover.png"}' \
  --out syntide-cover.pcraft

printf 'Built Syntide logos and cover in %s\n' "$asset_dir"
