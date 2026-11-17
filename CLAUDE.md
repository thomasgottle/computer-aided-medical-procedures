# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Course material for TUM "Computer Aided Medical Procedures 1 – AI in Computational Medicine & Surgery". Each exercise lives in its own folder; currently only `registration_practical/` exists. Its `README.md` and `data/README.md` are the authoritative descriptions of the exercise and data provenance — read them before changing behavior.

## Environment and running

All commands run from `registration_practical/`. Use Python 3.13 explicitly (a bare `python3` may be older):

```sh
python3.13 -m venv .venv313
./.venv313/bin/python -m pip install -r requirements.txt
./.venv313/bin/python -m jupyterlab CAMP1_registration_practical.ipynb
```

There is no build system, linter config, or test suite. To sanity-check helper changes, import the module and run the relevant functions directly, e.g.:

```sh
./.venv313/bin/python -c "import registration_lab as lab; c = lab.medical_intensity_case('ct_mr'); print(lab.register_intensity(c, 'MI'))"
```

The notebook's first setup cell auto-installs PyTorch and SimpleITK if missing and, on Colab, unpacks `CAMP1_registration_practical_student.zip`.

## Architecture

- **`registration_lab.py`** — pure NumPy/SciPy helper imported by the notebook as `lab`. Sections in file order: point ICP (`run_icp`, `rigid_fit`, `nearest_indices`), intensity registration (`register_intensity` with Powell over rigid params; SSD/NCC computed in NumPy, MI via SimpleITK Mattes, converted from negative nats to positive bits), deformable registration (`optimize_field` with explicit first-difference `regularizer`, `jacobian`), and medical cases (`medical_intensity_case('ct_ct'|'ct_mr')`, `medical_demo`). Plotting helpers sit next to each section.
- **`registration_learning.py`** — PyTorch VoxelMorph-style demo built on `registration_lab`. `TinyRegistrationNet` predicts a 5×5 control grid upsampled bicubically to a 128×128 field; `make_batch` generates pairs from a 3×3 random control field on real CT crops. `ARCHITECTURE_VERSION` guards against loading incompatible weights — bump it when changing the model or motion generator.
- **Notebooks** — `CAMP1_registration_practical.ipynb` (full student exercise, sections ICP → CT sampling → CT–CT → CT–MR → deformation → learning) and `CAMP1_registration_learning.ipynb` (standalone section 5 for Colab). The `.html` files are static exports of the student notebooks without outputs; regenerate them when notebooks change.
- **`data/`** — precomputed bundles (`.npz`, `.nii.gz`, `.hdf`) plus metadata JSON; students never download anything.

## Conventions that matter

- Coordinates are `(x, y)` but arrays are indexed `[y, x]`.
- Point ICP maps moving → fixed; image transforms map fixed → moving (backward sampling).
- Intensity metrics use source units with Gaussian smoothing only — no clipping, rescaling or histogram matching. Display windows affect plots only. Metrics use the full valid overlap including dark background; SSD is a sum, not a mean.
- Medical slices have 2 mm pixels but the optimizer works in pixel coordinates.
- Student editing tasks: `nearest_indices` and `metric_sign` have notebook counterparts where `None` triggers the reference fallback in `registration_lab`. Keep the helper's signatures compatible with those notebook hooks (e.g. `register_intensity(..., sign_function=...)`).
- Student notebooks must ship without saved outputs or answers; clear outputs before committing.
- The README references `instructor/` (solutions, `SESSION_GUIDE.md`, data-preparation scripts) and `results/`, which are intentionally not in this repo/student ZIP. `load_checkpoint`/`save_checkpoint` target `data/learning_weights_*.npz`, which are also not shipped — the notebook trains from scratch.
- Some numbers in `data/README.md` (e.g. ±8 px displacement bound) predate the current `coarse5x5_motion12_v2` constants in `registration_learning.py`; the code is authoritative.
