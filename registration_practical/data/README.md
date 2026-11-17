# Data provenance and interpretation

## CT–CT and CT–MR intensity experiments

`abdominal_ct_mr.npz` contains one real abdominal CT slice and the corresponding physical plane from the volume labelled MR in the previous course exercise:

`old version/camp-i-exercise-registration-main/intensity-based/SITK registration/data/{ct,mr}.nii.gz`

Both source volumes have size `(192, 160, 192)`, isotropic 2 mm spacing, origin `(0,0,0)` and identical direction matrices. The preparation script verifies all these properties before extracting axial index z=86 (physical z=172 mm). Identical indices correspond to the same physical plane **because geometry was verified**, not simply because the arrays have the same dimensions. No registration is fitted during preparation. Source hashes and full geometry are in `abdominal_ct_mr_metadata.json`. The bundled 2D arrays retain the source row/column order; no radiological display convention is implied.

`instructor/prepare_intensity_data.py` recreates this compact bundle from the original course volumes. SimpleITK is needed only for that preparation; students load NumPy arrays without additional dependencies or downloads.

The intensity-registration helper retains original CT and MR intensity units, with no clipping, rescaling or histogram matching. Both are smoothed with Gaussian sigma 0.65 pixels. CT window [-150,250] and the MR positive-intensity 99.5th-percentile range are used only by plotting functions; the metric arrays are unchanged. The CT–CT moving image is generated from the same preprocessed CT slice; the CT–MR moving image is generated from the preprocessed MR slice. Both receive an imposed backward-map rotation of 9 degrees and translation `(5,-4)` pixels about the image center. Bilinear resampling introduces small interpolation differences. The circular evaluation region has been removed: registration uses the full valid overlap including real dark background. Only out-of-image samples and generated padding are excluded.

The optimizer works in 2D pixel coordinates; one pixel corresponds to 2 mm. The displayed **imposed-motion error** is mean displacement error at five virtual probes. They are not manually annotated anatomical landmarks. For CT–CT this evaluates a known constructed transform. For CT–MR it measures recovery toward the supplied reference placement only. No independently validated CT–MR transform or anatomical landmarks were supplied. Shared geometry and visually similar organs do not prove true alignment; breathing and through-plane differences can remain. Original acquisition, patient relationship and licensing documentation were not supplied with these course files; the modality labels and source are inherited from the previous exercise.

## Synthetic controlled experiments

The point contour and deformable image pair are generated locally with fixed seeds. Known original point labels, fields and corresponding landmarks are used for evaluation and withheld from the fitting objectives. The small analytic intensity fixture remains in the helper solely for numerical regression tests; the student intensity activity uses medical slices.

## Breathing lung CT illustration

`fixed_slice.nii.gz` and `moving_slice.nii.gz` are copied unchanged from the same old course data folder. Both have size 550 by 550, spacing `(1,1)`, origin `(50,50)` and identity direction. SimpleITK smooths and downsamples them while preserving physical coordinates. Output geometry and stop conditions are recorded in `lung_metadata.json`.

`lung_demo.npz` stores fixed, identity-resampled moving and registered images on the downsampled fixed grid. `lung_transform.hdf` stores the fixed-to-moving physical-space transform, applying the B-spline map first and then the rigid map. `instructor/prepare_medical_demo.py` regenerates these outputs.

No ground-truth field or corresponding landmark set was supplied for this lung pair. Changes between slices may include out-of-plane motion. The B-spline example has no explicit lambda regularizer and is used only for optional qualitative discussion.

## Learning registration: CT crops and coarse motion

`learning_ct_slices.npz` contains 13 training crops and four held-out crops from the same supplied CT volume. Source slice indices are train 48–96 in increments of four and test 104,108,112,116. Crops use source rows 16:144 and columns 32:160, CT window [-150,250] scaled to [0,1], Gaussian sigma 0.8 source pixels, with no resizing or downsampling. Crops retain their native 128×128 pixels and 2 mm source spacing. The simplified network works in pixel coordinates. Generated motion amplitudes, model displacement bounds and ROI margins are doubled from the former 64×64 version to preserve approximately the same physical experiment.

Splitting source slices occurs before generating pairs. Training and evaluation use the same generator: a bicubically interpolated 3×3 random control field plus translation, smoothly bounded as `5*tanh(0.6*local + 0.4*translation)`. Control spacing is 63.5 source pixels over the 128×128 crop, producing broad regional motion. The moving image is the real CT crop, and the fixed image is its sampled version under the known backward field. That field is reserved for evaluation. The training function accepts only the images and predicted field in its loss. All anatomy comes from one supplied volume; held-out slices are not independent patients.

The small encoder/decoder has 11,322 parameters, takes two images, and predicts two components on a 5×5 control grid. Bicubic interpolation produces a dense field, followed by `8*tanh` to enforce the bound even when cubic interpolation overshoots. Predicted controls span the crop at 31.75-pixel spacing. These are coarse teaching deformations, not a biomechanical simulation; interpolation here is bicubic, not a cubic B-spline transform. Its architecture illustrates the VoxelMorph principle rather than reproducing the published architecture. The student notebook initializes weights from scratch, demonstrates one update, and then performs 1,000 additional live training updates. No saved weights are needed or included in the student ZIP.

The loss uses image MSE within a sixteen-pixel interior margin plus 0.01 times the mean squared neighboring field differences. The network bounds each displacement component to ±8 pixels, so image-loss samples remain in bounds. Training uses Adam at learning rate 0.001 and batches of eight generated training pairs after the single-pair demonstration. The last layer initializes near zero to start near identity. The same weights and optimizer continue through the entire exercise; inference freezes the model just trained.

`learning_metadata.json` records dataset preparation, source hash and coarse-field construction. `instructor/prepare_learning.py` prepares the CT crops and can produce offline reference runs; the student notebook never loads them. Reference weights from the earlier dense-output architecture must not be used with the new coarse-output model even though layer parameter shapes may coincide. Current validation results are in `instructor/coarse_learning_validation.json` and the executed notebook. Live student weights remain in memory unless explicitly saved.
