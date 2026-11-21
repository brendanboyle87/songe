import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe.models import worker


@pytest.mark.parametrize(
    ("source_type", "mlx_type"), [("qwen3_5_text", "qwen3_5"), ("llama", "llama")]
)
def test_mlx_conversion_uses_supported_model_type(
    tmp_path, monkeypatch, source_type, mlx_type
):
    source = tmp_path / "source"
    source.mkdir()
    config = {"model_type": source_type, "hidden_size": 64}
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    weights = source / "model.safetensors"
    weights.write_bytes(b"fixture weights")
    destination = tmp_path / "destination"
    destination.mkdir()

    def mlx_convert(*, hf_path, mlx_path, dtype):
        prepared = Path(hf_path)
        prepared_config = json.loads(
            (prepared / "config.json").read_text(encoding="utf-8")
        )
        assert prepared_config == config | {"model_type": mlx_type}
        assert (prepared / weights.name).samefile(weights)
        assert dtype == "bfloat16"
        output = Path(mlx_path)
        output.mkdir()
        (output / "config.json").write_text(
            json.dumps(prepared_config), encoding="utf-8"
        )
        (output / weights.name).write_bytes(weights.read_bytes())

    # This fixture only links files. Keep the fake macOS path testable on
    # Windows CI without requiring symlink privileges.
    if sys.platform == "win32":
        monkeypatch.setattr(Path, "symlink_to", Path.hardlink_to)
    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setitem(
        sys.modules, "mlx_lm.convert", SimpleNamespace(convert=mlx_convert)
    )
    worker.convert("mlx", source, destination)

    assert json.loads((source / "config.json").read_text(encoding="utf-8")) == config
    assert json.loads((destination / "config.json").read_text(encoding="utf-8")) == (
        config | {"model_type": mlx_type}
    )
    assert sorted(path.name for path in destination.iterdir()) == [
        "config.json",
        "model.safetensors",
    ]
