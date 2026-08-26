from pathlib import Path

import pytest

from perception_rt.audit import audit_dataset


def test_audit_dataset_rejects_nonpositive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Audit limit must be positive"):
        audit_dataset(tmp_path, limit=0)
