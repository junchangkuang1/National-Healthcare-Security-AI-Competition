# Inference

`scripts/infer.py` accepts one DICOM image, one NIfTI volume, or a directory that is searched recursively for both formats.

## Output

- DICOM segmentation masks: `<stem>_mask.png`
- NIfTI segmentation masks: `<stem>_mask.nii.gz`
- Prediction manifest: `predictions.json`

The JSON manifest records per-frame and mean-across-frame predictions for lesion presence and the auxiliary classification heads. TI-RADS, extension, and location predictions are converted back to one-based class identifiers. Malignancy and clinical-significance predictions remain binary (`0` or `1`).

## Important limitation

The exact submission schema and HTTP service contract were not present in the supplied source files. The included CLI provides deterministic local inference and generic exports, but it must be adapted to the official competition API specification before submission. In particular, confirm required filenames, JSON field names, label encodings, and whether DICOM masks must be returned in a format other than PNG.

