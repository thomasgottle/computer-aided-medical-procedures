# CAMP 1 registration practical

Open **CAMP1_registration_practical.ipynb** for the student exercise. Each numbered step separates the settings and result so students inspect a trial before changing a parameter or model. The notebook and HTML worksheet open without saved answers.

The sequence is ICP initialization → real CT sampling → translation versus rigid CT–CT registration → CT–MR metric selection → controlled deformation and regularization → learning registration → Q&A. There are no grading details or per-exercise time estimates.

## Setup

Use Python 3.13 explicitly; the package was tested with Python 3.13. A generic `python3` command may select an older interpreter even when Python 3.13 is installed. From this folder, create a separate environment and install the requirements:

```sh
python3.13 -m venv .venv313
./.venv313/bin/python --version
./.venv313/bin/python -m pip install -r requirements.txt
./.venv313/bin/python -m jupyterlab CAMP1_registration_practical.ipynb
```

The version check must show Python 3.13. If `python3.13` is not on your Mac PATH but the Python.org installation is present, use `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13` in the first command. An existing `.venv` made with an older Python version is not upgraded by activation; the new `.venv313` avoids reusing it.

On Windows, create the environment with `py -3.13 -m venv .venv313` and use `.venv313\Scripts\python.exe` instead of `./.venv313/bin/python` for the remaining commands. Open the student notebook and select the environment's Python kernel. In VS Code, select the same `.venv313` interpreter as the notebook kernel.

Keep this folder intact: the notebook imports `registration_lab.py` and reads the bundled `data/` directory. Run the setup cell to verify the environment before class. During the exercise, work cell by cell, predict each result and rerun trial cells with your chosen settings. Running all cells once produces only a first trial per core activity. Package installation needs internet access; running the lab does not need a GPU, online account, or downloads.

The setup cell prints the Python and numerical-library versions. A student who cannot execute the notebook can open **CAMP1_registration_practical.html** in a browser. It is a static worksheet without saved answers, so code edits will not run there. Completed outputs and figures are available separately to the instructor for demonstrations.

## After updating the practical

Rerun the setup cell after replacing the notebook or helper; it reloads `registration_lab.py` so an open kernel picks up new functions. If an older open notebook reports `AttributeError` for `medical_intensity_case`, run:

```python
import importlib
lab = importlib.reload(lab)
```

Then rerun the failed cell. If the attribute is still missing, inspect `lab.__file__` and use the complete updated practical folder at that location. Reinstalling packages is not needed for this error.

## Deep-learning section and free cloud execution

Section 5 adds the learning experiment to the full practical. **CAMP1_registration_learning.ipynb** contains the same five steps as a standalone notebook: untrained prediction, loss components, one explicit weight update, learning progress, and inference with frozen weights. The instructor has matching solutions and executed examples. It uses a small CPU network, 128×128 CT crops at native resolution, and full live training from scratch.

The first setup cell automatically installs PyTorch locally if needed. To install it manually instead:

```sh
./.venv313/bin/python -m pip install "torch>=2.2,<3"
```

For Google Colab:

1. Open [Google Colab](https://colab.research.google.com/) and upload `CAMP1_registration_learning.ipynb` (or the full practical notebook).
2. Use the default CPU runtime. Run the first code cell and upload `CAMP1_registration_practical_student.zip` when prompted.
3. Proceed one step at a time. The ZIP supplies the helpers and data; no Drive mount or remote model download is needed. Colab normally includes PyTorch, NumPy, SciPy and Matplotlib. If PyTorch is missing, the notebook provides a `%pip` command.

On a local machine, the first setup cell checks for PyTorch and installs it only if missing, using the notebook kernel's Python executable. Free cloud resources are not guaranteed; the same notebook can run locally on CPU. The full practical's local version pins remain in `requirements.txt`; do not reinstall those pins over Colab's existing stack unnecessarily.

Step 5.1 initializes fresh weights. Step 5.3 performs one demonstration update. Step 5.4 then runs 1,000 additional updates using batches of newly generated training pairs, for 1,001 total updates when run in order. Progress is printed every 100 updates. No checkpoint is loaded and no checkpoint selector is present. Step 5.5 uses the model just trained. Rerunning step 5.4 adds another 1,000 updates; rerunning step 5.1 resets the model. Live weights remain in memory and are not automatically written to disk.

## Student editing tasks

The first short task implements nearest-neighbor correspondence indices. The second chooses the minimization sign for SSD, signed NCC and MI. In both tasks, `None` activates a working reference fallback. Replace the `None` values to use your own answer. The remaining work consists of changing experimental settings and explaining plots and numerical results.

For ICP, investigate initialization before changing the iteration budget. For CT–CT, inspect manual sampling, run translation only, then add rotation and test initialization. For CT–MR, inspect the new pair before running a baseline metric, then change only the metric. For deformation, inspect the rigid baseline and unregularized field before choosing regularization. Inspect each result before the next step.

Rerun a settings cell and its immediately following result cell to try a new configuration. Logs retain previous choices and measurements. Input/setup cells clear their own logs. Do not use Run All as the exercise workflow; it bypasses the prediction and inspection stages.
The instructor notebook retains prepared comparisons and worked explanations. Its separate executed student-notebook copy records the validation run. Neither is included in the student ZIP.

## Files

| File or folder | Purpose |
| --- | --- |
| `CAMP1_registration_practical.ipynb` | Student notebook with reference fallbacks and empty trial logs |
| `CAMP1_registration_practical.html` | Static worksheet without answer outputs |
| `registration_lab.py` | Inspectable NumPy/SciPy implementation and plotting helpers |
| `registration_learning.py` | Small PyTorch network, differentiable sampler and image-based training |
| `CAMP1_registration_learning.ipynb` | Standalone learning section for local execution or Colab |
| `data/` | CT/MR plane bundle, learning CT crops, metadata, and supporting image assets |
| `results/` | Instructor reference results and figure fallbacks; omitted from the student ZIP |
| `instructor/CAMP1_registration_practical_solutions.ipynb` | Completed code and explanations |
| `instructor/CAMP1_registration_practical_solutions.html` | Static instructor solutions |
| `instructor/SESSION_GUIDE.md` | Facilitation notes, expected results and regeneration commands |

Distribute the student ZIP, which includes the notebook, helper, requirements, static worksheet and data. Keep the `instructor/` and `results/` folders separate when sharing the exercise. The supplied reference helper is intentionally readable; the activity focuses on interpreting registration behavior.

## Interpretation

The point and deformation cases use unit pixel spacing. The medical intensity slices use 2 mm pixels, and the simplified optimizer works in pixel coordinates. Point ICP maps moving points to fixed points; image resampling uses a fixed-to-moving backward map.

CT–CT has a known imposed transform. CT–MR uses a supplied reference placement without independently validated anatomical correspondence. Its displayed imposed-motion error is a recovery diagnostic, not anatomical registration accuracy. Virtual evaluation probes are not manually annotated landmarks. See `data/README.md` for source geometry and preparation.

Intensity metrics use the full valid overlap, including dark background; no circular or anatomical ROI is applied. The deformable experiment uses a fixed interior region, an explicit first-difference regularizer and a fixed optimizer budget. Sampled Jacobians do not prove global invertibility.

Implementation references: [SciPy's Powell optimizer](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html) and [SimpleITK registration conventions](https://simpleitk.readthedocs.io/en/main/registrationOverview.html).

Deformable-registration results show the target, moving and registered images, both overlays, the sampling grid, signed differences to the target before and after registration, and the Jacobian map. The paired difference maps share limits anchored to the unregistered pair; they are not normalized independently. Gray marks the region excluded from evaluation or unavailable samples.

The learning experiment uses broad generated motion from a 3×3 control grid. The network predicts a 5×5 grid and smoothly interpolates it to the 128×128 sampling field. This constrains deformation at the model level; the displayed field is the same field used by the sampler and loss. Training still starts from scratch with one demonstration update followed by 1,000 further updates.

The current learning version is `coarse5x5_motion12_v2`. It increases generated motion amplitude 2.4-fold while preserving the broad 3×3 control grid. Step 5.1 prints the loaded helper path, version and actual displacement magnitude. After updating, rerun step 5.1 to recreate the image pair and fresh network, then the one-update and 1,000-update cells. Reloading a module alone leaves old image pairs and model objects in memory. On Colab, use the updated ZIP in a fresh runtime if old helper files are still present.

MI registration and objective plots use SimpleITK Mattes mutual information (32 bins, Parzen-window estimation, all pixels in the valid image overlap). SimpleITK negative MI in nats is converted to positive MI in bits. The Powell optimizer and bounded search are shared with SSD and NCC. The first setup cell installs SimpleITK if missing, including on Colab.

The CT–CT and CT–MR intensity exercises use source intensity units, with Gaussian smoothing only. There is no intensity rescaling, clipping or histogram matching before registration. Plot windows affect display only. With source intensities and full valid overlap, the default CT–MR SSD run misaligns while Mattes MI recovers the imposed motion more closely; this is still not an anatomical ground-truth evaluation. The separate learning-based CT–CT experiment retains its documented network-input preprocessing.

Medical intensity registration compares every pixel with a valid moving-image interpolation footprint. Real background is included. Only out-of-image samples and artificial motion padding are excluded. SSD is the sum, not the mean, of squared differences; the overlap size can change with the transform.
