"""Verified single-writer checkpoint snapshots for a mounted persistent drive."""
import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(root):
    return {
        str(file.relative_to(root)): [file.stat().st_size, sha256(file)]
        for file in sorted(root.rglob("*"))
        if file.is_file() and file != root / "manifest.json"
    }


def complete(path):
    required = (
        "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
        "scheduler.pt", "trainer_state.json", "rng_state.pth",
    )
    try:
        state = json.loads((path / "trainer_state.json").read_text())
        return (
            path.name == f"checkpoint-{state['global_step']}"
            and all((path / name).stat().st_size > 0 for name in required)
        )
    except (OSError, ValueError, KeyError):
        return False


def snapshots(root):
    return sorted(
        (p for p in root.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]), reverse=True,
    )


def verified(path, signature):
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["files"] != inventory(path) or not complete(path / path.name):
            return False
    except (OSError, ValueError, KeyError):
        return False
    if manifest["run"] != signature:
        raise ValueError(f"Configuração diferente do checkpoint {path}; use outro diretório de execução.")
    return True


def publish(root, checkpoint, best, signature):
    if not complete(checkpoint):
        raise RuntimeError(f"Checkpoint incompleto: {checkpoint}")
    if not all((best / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors")):
        raise RuntimeError(f"Melhor checkpoint incompleto: {best}")
    destination = root / checkpoint.name
    if destination.exists() and verified(destination, signature):
        return
    with TemporaryDirectory(prefix=".staging-", dir=root) as directory:
        stage = Path(directory) / checkpoint.name
        stage.mkdir()
        expected = {f"{checkpoint.name}/{key}": value for key, value in inventory(checkpoint).items()}
        shutil.copytree(checkpoint, stage / checkpoint.name)
        best_adapter = stage / "best_adapter"
        best_adapter.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            source = best / name
            expected[f"best_adapter/{name}"] = [source.stat().st_size, sha256(source)]
            shutil.copy2(source, best_adapter / name)
        (stage / "manifest.json").write_text(json.dumps({
            "files": expected, "run": signature, "best": best.name,
        }, indent=2))
        if not verified(stage, signature):
            raise RuntimeError(f"Falha ao verificar {stage}")
        if destination.exists():
            quarantine = root / f".corrupt-{destination.name}-{time.time_ns()}"
            destination.rename(quarantine)
            logging.warning("Snapshot corrompido isolado em %s", quarantine)
        stage.rename(destination)
    for old in [path for path in snapshots(root) if verified(path, signature)][3:]:
        shutil.rmtree(old)


def restore(root, output, signature):
    for path in snapshots(root):
        if not verified(path, signature):
            logging.warning("Ignorando snapshot corrompido: %s", path)
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        destination = output / path.name
        with TemporaryDirectory(prefix=".restore-", dir=output) as directory:
            stage = Path(directory) / path.name
            shutil.copytree(path / path.name, stage)
            best = output / manifest["best"]
            if best != destination:
                best.mkdir(exist_ok=True)
                shutil.copytree(path / "best_adapter", best, dirs_exist_ok=True)
            state_path = stage / "trainer_state.json"
            state = json.loads(state_path.read_text())
            state["best_model_checkpoint"] = str(best)
            state_path.write_text(json.dumps(state))
            if destination.exists():
                shutil.rmtree(destination)
            stage.rename(destination)
        return str(destination)
    if snapshots(root):
        raise RuntimeError("Nenhum snapshot íntegro encontrado.")
    return None
