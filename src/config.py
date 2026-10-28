"""Localiza la raíz del proyecto y carga la configuración (small / large)."""
from pathlib import Path
import yaml


def find_root(start=None) -> Path:
    """Sube carpetas hasta encontrar requirements.txt."""
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "requirements.txt").exists():
            return p
    raise FileNotFoundError("No se encontró la raíz del proyecto (requirements.txt).")


ROOT = find_root()


def load_config(name: str = "small") -> dict:
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Rutas locales -> absolutas; rutas s3:// se dejan como texto (para EMR)
    cfg["paths"] = {k: (v if str(v).startswith("s3") else ROOT / v)
                    for k, v in cfg["paths"].items()}
    return cfg