"""
BSORT: Bottle Cap Sorting CLI Tool.
This module provides a command-line interface for training, inferring,
and exporting YOLO models.
"""
import yaml
import typer
from pathlib import Path
from ultralytics import YOLO
import wandb

app = typer.Typer(help="BSORT: Bottle Sorting AI CLI Tool")

def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@app.command()
def train(config: str = typer.Option(..., "--config", help="Path to settings.yaml")):
    """Trains the YOLO model based on configuration."""
    cfg = load_config(config)
    typer.echo(f"🚀 Starting training with config: {config}")
    
    wandb.init(project=cfg.get("project_name", "bsort"), config=cfg)
    model = YOLO(cfg["model"]["architecture"])

    model.train(
        data=cfg["train"]["data_path"],
        epochs=cfg["train"]["epochs"],
        batch=cfg["train"]["batch_size"],
        imgsz=cfg["model"]["imgsz"],
        patience=cfg["train"].get("patience", 30),
        device=cfg["model"]["device"],
        project=cfg.get("project_name"),
        name="cli-run",
        verbose=True
    )
    typer.echo("✅ Training complete!")

@app.command()
def infer(
    config: str = typer.Option(..., "--config", help="Path to settings.yaml"),
    image: str = typer.Option(..., "--image", help="Path to image file"),
    use_onnx: bool = typer.Option(False, help="Use exported ONNX model")
):
    """Runs inference on a single image."""
    cfg = load_config(config)
    base_path = Path(cfg["model"]["weights_path"])
    
    if use_onnx:
        model_path = str(base_path.with_suffix('.onnx'))
        typer.echo(f"⚡ Using ONNX Optimized Model: {model_path}")
    else:
        model_path = str(base_path)
        typer.echo(f"🐢 Using Standard PyTorch Model: {model_path}")

    try:
        model = YOLO(model_path)
        results = model.predict(
            source=image,
            imgsz=cfg["model"]["imgsz"],
            conf=cfg["inference"]["conf_threshold"],
            device=cfg["model"]["device"],
            save=True
        )
        typer.echo(f"📄 Result saved to: {results[0].save_dir}")
    except Exception as e:
        typer.echo(f"❌ Error during inference: {e}")

@app.command()
def export(
    config: str = typer.Option(..., "--config", help="Path to settings.yaml"),
    format: str = typer.Option("onnx", help="Target format: onnx, ncnn"),
    half: bool = typer.Option(True, help="Use FP16 quantization")
):
    """Exports the model for deployment (RPi)."""
    cfg = load_config(config)
    model_path = cfg["model"]["weights_path"]
    
    typer.echo(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    typer.echo(f"⚙️ Exporting to {format.upper()} (FP16={half})...")
    path = model.export(format=format, half=half, imgsz=cfg["model"]["imgsz"])
    typer.echo(f"✅ Export success! Model saved at: {path}")

if __name__ == "__main__":
    app()