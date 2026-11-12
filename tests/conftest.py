"""Cho phép chạy pytest từ gốc kho trên Windows và Colab."""
from pathlib import Path
import sys

import pytest

GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))


@pytest.fixture(scope="session")
def bang():
    from ml.du_lieu import doc_so_lieu
    return doc_so_lieu()
