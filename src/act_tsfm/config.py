from pathlib import Path
import hashlib
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load(path: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or ROOT / 'configs/experiment.yaml').read_text())
    cfg.pop('hash',None)
    return cfg

def directories() -> None:
    for p in ['data/raw','data/processed','data/manifests','results/tables','results/figures',
              'results/logs','results/manifests','artifacts/oracle_reference','artifacts/observed_losses','artifacts/checkpoints']:
        (ROOT / p).mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
