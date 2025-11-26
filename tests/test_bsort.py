import pytest
from pathlib import Path
from bsort.main import load_config

def test_load_config_success(tmp_path):
    """Ensure valid config file loads correctly."""
    d = tmp_path / "settings.yaml"
    d.write_text("project_name: test\nmodel:\n  architecture: yolo.pt")
    
    cfg = load_config(str(d))
    assert cfg["project_name"] == "test"

def test_load_config_not_found():
    """Ensure FileNotFoundError is raised for missing config."""
    with pytest.raises(FileNotFoundError):
        load_config("ghost_file.yaml")