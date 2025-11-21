"""
Skeleton training script for Flood Risk Prediction.
Keep heavy imports inside functions to allow safe syntax-checking without installing all deps.
"""

import argparse
import os
import sys

# Ensure repo root is in sys.path so `import src.*` works when running this script directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate flood risk model (skeleton)")
    parser.add_argument("--data-dir", type=str, default="data/", help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--model-out", type=str, default="model.h5", help="Path to save trained model")
    parser.add_argument("--demo", action="store_true", help="Run a small demo using synthetic data")
    parser.add_argument("--model-name", type=str, default="small", help="Model backbone name: small|resnet50")
    return parser.parse_args()


def preprocess_data(data_dir, img_size=(224, 224), batch_size=16, demo=False):
    """Dataset loader placeholder. If demo=True, generate a synthetic tf.data.Dataset pair.
    """
    if demo:
        try:
            from src.data import get_dataset
            train_ds, val_ds = get_dataset(mode="synthetic", num_samples=32, img_size=img_size, batch_size=batch_size)
            return train_ds, val_ds
        except Exception as e:
            print('Could not create synthetic dataset:', e)
            return None, None

    print(f"Preprocessing data from {data_dir} — img_size={img_size}. (This is a placeholder for real loaders.)")
    return None, None


def build_model(input_shape=(224, 224, 3), num_classes=3, model_name="small"):
    try:
        from src.model import build_model as _build
    except Exception as e:
        print('Model builder import failed:', e)
        return None

    try:
        return _build(name=model_name, input_shape=input_shape, num_classes=num_classes)
    except Exception as e:
        print('Failed to build model:', e)
        return None


def train(args):
    img_size = (128, 128, 3)
    train_ds, val_ds = preprocess_data(args.data_dir, img_size=img_size, batch_size=args.batch_size, demo=args.demo)
    model = build_model(input_shape=img_size, num_classes=3, model_name=args.model_name)
    if model is None:
        print("Model could not be built (missing dependencies). Install tensorflow and re-run.")
        return

    if train_ds is None:
        print("No dataset available. Use --demo to run on a synthetic dataset or implement real data loader.")
        return

    print("Starting training")
    try:
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    except Exception as e:
        print('Training failed or was skipped:', e)

    try:
        model.save(args.model_out)
        print(f"Model saved to {args.model_out}")
    except Exception as e:
        print('Model save failed (maybe running on TF missing filesystem support):', e)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
