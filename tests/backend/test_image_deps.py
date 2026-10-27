"""The backend image installs no ``ml`` dependency group (LightGBM, MLflow, Optuna)."""

import subprocess
import sys

BLOCK_ML_GROUP = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"lightgbm", "mlflow", "optuna"}:
            raise ImportError("not in the backend image: " + name)
sys.meta_path.insert(0, Block())
import backend.app.main
"""


def test_backend_imports_without_the_ml_group() -> None:
    result = subprocess.run(
        [sys.executable, "-c", BLOCK_ML_GROUP], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
