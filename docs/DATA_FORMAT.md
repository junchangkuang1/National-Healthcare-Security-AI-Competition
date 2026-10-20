# Data format

## Supported inputs

The preprocessing pipeline supports two acquisition types:

1. **Static ultrasound** — one or more DICOM (`.dcm`) images in a series directory, with optional lesion masks stored as NIfTI files.
2. **Dynamic ultrasound** — a NIfTI (`.nii` or `.nii.gz`) volume whose third dimension is time/frame, accompanied by one or more NIfTI lesion masks.

Mask names are recognized when they contain `_mask`, `mask_`, `病灶`, or `主病灶`, or begin with `M_`/`M<digit>`. Adapt `MASK_HINTS` in `src/thyroidxl/data.py` if the local release uses a different convention.

Example only:

```text
annotation/
├── clinical_labels.xlsx
├── patient_001/
│   ├── static_series/
│   │   ├── image.dcm
│   │   └── M_mask.nii.gz
│   └── dynamic_series/
│       ├── dynamic_series.nii.gz
│       └── 病灶1_mask.nii.gz
└── patient_002/
    └── ...
```

## Clinical table

The Excel parser searches the first 20 rows of each worksheet for a header containing a patient identifier and a series/accession identifier. It recognizes aliases for:

- patient ID;
- series UID;
- accession number;
- ROI name;
- malignancy;
- TI-RADS class;
- extrathyroidal extension;
- lesion location;
- clinical T-stage significance.

Missing auxiliary labels are encoded with an ignore index and do not contribute to their cross-entropy loss.

## Split policy

Patients, rather than frames or images, are shuffled with a fixed seed and divided into approximately 80% training, 10% validation, and 10% held-out testing. The generated split file should be retained with the experiment artifacts but must not be committed when it contains real patient identifiers.

## Privacy

The generated JSONL index contains source paths and patient identifiers. It is therefore written under `metadata/`, ignored by Git, and should be handled according to the dataset agreement and institutional privacy policy.

