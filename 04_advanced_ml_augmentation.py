from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "04_advanced_ml_augmentation.py"), run_name="__main__")