# Comic Panel Detector

Detect and extract individual panels from scanned comic pages using classical computer‑vision techniques (no deep learning required). The script combines three complementary strategies—contour analysis, watershed segmentation, and a projection‑based grid fallback—to give robust results across a wide range of art styles, page layouts, and scan qualities.

---

## ✨ Key features

* **Multiple detection modes**
  * **Contour method** – configurable black/white/dual edge masks or Canny detection.
  * **Watershed method** – automatically segments large connected regions that were missed by contours.
  * **Grid fallback** – projection‑profile analysis for very regular layouts.
* **Automatic de‑duplication & sorting** – merges overlapping results, then orders panels row‑by‑row (western reading order by default).
* **High‑quality crops & annotated previews** – saves each panel as a JPEG/PNG and writes an optional metadata JSON file.
* **Rich debugging output** – dump intermediate masks, contours, markers, and more with `--debug` or visualise them live with `--debug‑viz`.
* **Zero learning curve** – one Python file, no training data, no GPU.

---

## Installation

```bash
# 1. Clone the repo (or drop the script into your own project)
git clone https://github.com/<your‑user>/comic‑panel‑detector.git
cd comic‑panel‑detector

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt  # OpenCV, NumPy, SciPy, scikit‑image, etc.
```

> **Python ≥ 3.8** is required.  All heavy lifting happens in **OpenCV**; no deep‑learning frameworks are needed.

---

## Quick start

```bash
python comic_panel_detector.py page.jpg -o output/ --json --crops
```

The command above will:

1. Read **`page.jpg`**.
2. Detect panels with the default dual‑edge strategy.
3. Write **`output/page_annotated.jpg`** (original page + bounding boxes).
4. Save every panel crop as **`output/page_panel_XX.jpg`**.
5. Dump panel metadata to **`output/page_panels.json`**.

Open the annotated page to see how the detector performed.  If some panels are missing or split, read on for tuning tips.

---

## Command‑line options (essentials)

| Flag | Default | Purpose |
|------|---------|---------|
| `--edge-method {white,black,dual,canny}` | `dual` | Which contour mask(s) to build. |
| `--black-threshold INT` | `60` | Threshold for *black* line mask (0‒255). |
| `--white-threshold INT` | `200` | Threshold for *white* gutter mask (0‒255). |
| `--min-area-ratio FLOAT` | `0.01` | Smallest panel area w.r.t. full page. |
| `--max-area-ratio FLOAT` | `0.5` | Largest panel area w.r.t. full page. |
| `--line-thickness-ratio FLOAT` | `0.005` | Kernel size used when closing/dilating masks. |
| `--morph-iterations INT` | `2` | How many times to dilate/erode. |
| `--pad FLOAT` | `0.05` | Extra padding (5 % by default) around panel crops. |
| `--no-contour / --no-watershed / --no-grid` | – | Disable individual detection stages. |
| `--debug` | – | Write every intermediate image to disk. |
| `--debug-viz` | – | Pop up interactive windows for on‑the‑fly inspection. |
| `--json` | – | Export results to a machine‑readable JSON file. |
| `--no-crops` | – | Skip cropping (useful when only the metadata is needed). |

---

## Fine‑tuning & troubleshooting

* **Over‑segmentation (page split into too many panels)** – raise `--min-area-ratio` or tighten `--white-threshold` so thin gutters are ignored.
* **Missed panels** –
  * Lower `--min-area-ratio` if tiny insets are expected.
  * Increase `--max-area-ratio` for full‑bleed splash pages.
  * Try `--edge-method canny` when line art is faint or non‑uniform.
* **Watershed artefacts** – you can disable the stage with `--no-watershed` or tweak `--morph-iterations`.
* **Regular grids** (newspaper strips, manga with strict layout) – the projection fallback usually nails these; ensure `--no-grid` is *not* set.

---

## Output files

```
output/
├── page_annotated.jpg      # original + bounding boxes
├── page_panel_01.jpg       # cropped panel (padding applied)
├── page_panel_02.jpg
├── page_panels.json        # metadata (bbox, area, method, etc.)
└── ...
```

The JSON structure is a list of panel dictionaries:

```json
[
  {
    "bbox": [x, y, w, h],
    "area": 123456,
    "method": "contour",  // or "watershed" / "grid"
    "aspect_ratio": 1.25,
    "area_ratio": 0.82
  },
  ...
]
```

---

## API usage

If you want to call the detector from another Python script instead of the CLI:

```python
from pathlib import Path
from comic_panel_detector import detect_panels_improved, draw_panels

img, panels, debug = detect_panels_improved(
    Path("page.jpg"),
    Path("output"),
    edge_method="dual",
    debug=False
)

annotated = draw_panels(img, panels)
```

The helper returns `(img_bgr, panels, debug_images)` so you can further post‑process or visualise as needed.

---

## Contributing

Pull requests are very welcome!  Please open an issue first to discuss major changes.  Make sure your code passes **`ruff`**/`flake8` and type checks with **`mypy`**.  New features should come with unit tests (see `tests/`).

---

## License

This project is released under the **MIT License**—see [LICENSE](LICENSE) for details.

---

## Acknowledgements

* Built with ❤️ on top of **OpenCV**, **NumPy**, **SciPy**, and **scikit‑image**.
* Inspired by the many open‑source efforts tackling comic‑page analysis.

> Happy hacking & enjoy your panelised comics! 📚🎉

