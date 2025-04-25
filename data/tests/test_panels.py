from pathlib import Path
import numpy as np

from panel_detector import detect_panels_improved

TEST_IMG = Path("tests/data/simple_page.jpg")  # add a small sample image

def test_detects_expected_panels(tmp_path):
    """The detector should find exactly 3 panels on the sample page."""
    img, panels, _ = detect_panels_improved(
        TEST_IMG,
        tmp_path,          # tmp output dir (pytest creates it)
        enable_grid=True,  # rely on fast grid fallback for this image
        debug=False
    )

    assert len(panels) == 3

    # basic sanity checks on first panel bbox
    x, y, w, h = panels[0]["bbox"]
    H, W = img.shape[:2]
    assert 0 <= x < W and 0 <= y < H
    assert w > 0 and h > 0
