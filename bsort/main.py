"""
BSORT: Bottle Cap Sorting CLI Tool.

This module provides a command-line interface for training, inferring,
and exporting YOLO models for bottle cap detection.
"""

from pathlib import Path

import typer
import wandb
import yaml
from ultralytics import YOLO

app = typer.Typer(help="BSORT: Bottle Sorting AI CLI Tool")


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.command()
def train(config: str = typer.Option(..., "--config", help="Path to settings.yaml")):
    """Trains the YOLO model based on configuration.

    Initializes WandB for tracking and starts the training process
    using parameters defined in the YAML config.

    Args:
        config (str): Path to configuration file.
    """
    cfg = load_config(config)
    typer.echo(f"🚀 Starting training with config: {config}")

    # 1. Init WandB
    wandb.init(project=cfg.get("project_name", "bsort"), config=cfg)

    # 2. Load Model
    model = YOLO(cfg["model"]["architecture"])

    # 3. Start Training
    train_args = cfg["train"]
    model.train(
        data=train_args["data_path"],
        epochs=train_args["epochs"],
        batch=train_args["batch_size"],
        imgsz=cfg["model"]["imgsz"],
        patience=train_args.get("patience", 30),
        device=cfg["model"]["device"],
        project=cfg.get("project_name"),
        name="cli-run",
        verbose=True,
    )
    typer.echo("✅ Training complete!")


@app.command()
def infer(
    config: str = typer.Option(..., "--config", help="Path to settings.yaml"),
    image: str = typer.Option(..., "--image", help="Path to image file"),
    use_onnx: bool = typer.Option(False, help="Use exported ONNX model for speed"),
):
    """Runs inference on a single image.

    Args:
        config (str): Path to configuration file.
        image (str): Path to input image.
        use_onnx (bool): If True, looks for .onnx model instead of .pt.
    """
    cfg = load_config(config)

    # Logic pilih model (.pt atau .onnx)
    base_path = Path(cfg["model"]["weights_path"])

    if use_onnx:
        model_path = str(base_path.with_suffix(".onnx"))
        typer.echo(f"⚡ Using ONNX Optimized Model: {model_path}")
    else:
        model_path = str(base_path)
        typer.echo(f"🐢 Using Standard PyTorch Model: {model_path}")

    # Load & Predict
    try:
        model = YOLO(model_path)
        results = model.predict(
            source=image,
            imgsz=cfg["model"]["imgsz"],
            conf=cfg["inference"]["conf_threshold"],
            device=cfg["model"]["device"],
            save=True,
        )
        typer.echo(f"📄 Result saved to: {results[0].save_dir}")
    # pylint: disable=broad-exception-caught
    except Exception as e:
        typer.echo(f"❌ Error during inference: {e}")


@app.command()
def export(
    config: str = typer.Option(..., "--config", help="Path to settings.yaml"),
    export_format: str = typer.Option(
        "onnx", "--format", help="Target format: onnx, ncnn, tflite"
    ),
    half: bool = typer.Option(True, help="Use FP16 Half-Precision (Recommended)"),
):
    """Exports the model for deployment on Edge Devices (RPi).

    Args:
        config (str): Path to configuration file.
        export_format (str): Target export format. Defaults to 'onnx'.
        half (bool): Enable FP16 quantization for speedup.
    """
    cfg = load_config(config)
    model_path = cfg["model"]["weights_path"]

    typer.echo(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)

    typer.echo(f"⚙️ Exporting to {export_format.upper()} (FP16={half})...")
    path = model.export(format=export_format, half=half, imgsz=cfg["model"]["imgsz"])
    typer.echo(f"✅ Export success! Model saved at: {path}")


if __name__ == "__main__":
    app()
