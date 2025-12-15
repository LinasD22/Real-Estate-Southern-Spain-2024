import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model import load_preprocessed_data


if __name__ == "__main__":
    # Change to project root directory for relative paths to work correctly
    import os
    os.chdir(project_root)

    X_test = load_preprocessed_data().get("X_tr")
    print(X_test)