import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage.data.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    return generate_synthetic_dataset(root, n_clips=80, num_frames=8, image_size=32, seed=1)
