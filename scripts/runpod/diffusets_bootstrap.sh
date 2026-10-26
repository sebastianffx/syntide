#!/usr/bin/env bash
# Separate generation/trainer venvs; retain the CUDA image's torch wheels.
set -euo pipefail
phase="${1:-generation}"
mkdir -p /workspace/diffusets-study/setup /workspace/baseline-runtime
python3 - <<'PY'
import json, pathlib, time
s=json.loads(pathlib.Path('/workspace/diffusets-extension/guard-state.json').read_text())
assert s['status']=='armed' and s['pod_id']=='0il9463gvr3mmv'
assert pathlib.Path('/proc',str(s['guard_pid']),'cmdline').exists()
assert time.time()-pathlib.Path('/workspace/diffusets-extension/guard-state.json').stat().st_mtime<180
PY
if [ "$phase" = generation ]; then
  python3 -m venv --system-site-packages /workspace/diffusets-generation-venv
  python3 - <<'PY'
import importlib.metadata, pathlib
pathlib.Path('/workspace/diffusets-study/setup/image-torch-constraints.txt').write_text(''.join(
    f'{p}=={importlib.metadata.version(p)}\n' for p in ('torch','torchvision','torchaudio')))
PY
  /workspace/diffusets-generation-venv/bin/python -m pip install --no-cache-dir \
    -c /workspace/diffusets-study/setup/image-torch-constraints.txt \
    diffusers==0.35.1 numpy==2.2.6 scipy==1.16.2 pandas==2.3.3 \
    huggingface-hub==0.35.3 wfdb==4.3.0 pyarrow==21.0.0 pyyaml==6.0.2 \
    timenet==0.11.0 timenet-connectors==0.11.0 hydra-core==1.3.2
  PYTHONPATH=/workspace/diffusets-code/src /workspace/diffusets-generation-venv/bin/python - <<'PY'
import hashlib, importlib.metadata, json, pathlib, sys, torch, wfdb
from syntide.generate.full.assets import CHECKPOINT_HASHES, SOURCE_HASHES
from syntide.generate.full.diffusets import DiffuSETSBackend
from syntide.generate.qa.evidence import EvidenceExtractor
assert torch.__version__=='2.8.0.dev20250319+cu128' and torch.cuda.is_available()
assets=pathlib.Path('/workspace/diffusets-assets')
for name, expected in CHECKPOINT_HASHES.items():
    with (assets/'checkpoints'/name).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected, name
for name, expected in SOURCE_HASHES.items():
    assert hashlib.sha256((assets/'source'/name).read_bytes()).hexdigest()==expected,name
result={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'packages':{p:importlib.metadata.version(p)
        for p in ['numpy','scipy','pandas','diffusers','huggingface-hub','wfdb','pyarrow']},
        'pinned_assets_verified':True,'generation_started':False}
pathlib.Path('/workspace/diffusets-study/setup/generation-environment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
  /workspace/diffusets-generation-venv/bin/python -m pip freeze > /workspace/diffusets-study/setup/generation-pip-freeze.txt
elif [ "$phase" = baseline ]; then
  tar --no-same-owner --no-same-permissions -xzf \
    /workspace/baseline-runtime/setup/source_snapshot.tar.gz -C /workspace/baseline-runtime
  python3 -m venv --system-site-packages /workspace/baseline-runtime/venv
  /workspace/baseline-runtime/venv/bin/python -m pip install --no-cache-dir \
    -c /workspace/diffusets-study/setup/image-torch-constraints.txt \
    -r /workspace/baseline-runtime/scripts/runpod/requirements.txt
  /workspace/baseline-runtime/venv/bin/python -m pip install --no-cache-dir --no-deps open-flamingo==2.0.1
  if [ ! -d /workspace/baseline-runtime/OpenTSLM/.git ]; then
    git clone --filter=blob:none https://github.com/OpenTSLM/OpenTSLM.git /workspace/baseline-runtime/OpenTSLM
  fi
  git -C /workspace/baseline-runtime/OpenTSLM checkout 2968f4b891baab4307f7e9d0043e87677b593a30
  PYTHONPATH=/workspace/baseline-runtime/src:/workspace/baseline-runtime/trainers/src:/workspace/baseline-runtime/OpenTSLM/src \
    /workspace/baseline-runtime/venv/bin/python - <<'PY'
import json, pathlib, subprocess, torch
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
assert torch.__version__=='2.8.0.dev20250319+cu128' and torch.cuda.is_available()
revision=subprocess.check_output(['git','-C','/workspace/baseline-runtime/OpenTSLM','rev-parse','HEAD'],text=True).strip()
assert revision=='2968f4b891baab4307f7e9d0043e87677b593a30'
result={'torch':torch.__version__,'cuda':torch.version.cuda,'opentslm_commit':revision,'import_ok':True}
pathlib.Path('/workspace/diffusets-study/setup/baseline-environment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
  /workspace/baseline-runtime/venv/bin/python -m pip freeze > /workspace/diffusets-study/setup/baseline-pip-freeze.txt
else
  echo 'Unknown bootstrap phase' >&2
  exit 2
fi
