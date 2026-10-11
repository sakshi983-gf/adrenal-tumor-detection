"""
convert_patient.py — Prepare 22-slice test series for Patient Adrenal_Ki67_Seg_041
Supports either a raw DICOM directory (if present) or the NIfTI volume in dataset/ct_volumes/.
Saves 22 windowed CT slices [-160, 240] HU to test_data/patient_041/slice_000.png ... slice_021.png
"""

import os
import cv2
import numpy as np

DICOM_DIR = "Adrenal-ACC-Ki67-Seg/Adrenal_Ki67_Seg_041/CT_series/"
NIFTI_PATH = "dataset/ct_volumes/Adrenal_Ki67_Seg_041/Adrenal_Ki67_Seg_041_original_ct.nii.gz"
OUTPUT_DIR = "test_data/patient_041/"


def convert_patient_041():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Option 1: Convert from DICOM directory if available locally
    if os.path.isdir(DICOM_DIR):
        import pydicom

        files = sorted([f for f in os.listdir(DICOM_DIR) if f.endswith(".dcm")])
        for i, filename in enumerate(files[:22]):
            ds = pydicom.dcmread(os.path.join(DICOM_DIR, filename))
            pixel = ds.pixel_array.astype(np.float32)
            hu = pixel * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
            windowed = np.clip(hu, -160, 240)
            normalized = ((windowed + 160) / 400.0 * 255.0).astype(np.uint8)
            out_path = os.path.join(OUTPUT_DIR, f"slice_{i:03d}.png")
            cv2.imwrite(out_path, normalized)
        print(f"Saved {min(len(files), 22)} PNG slices from DICOM to {OUTPUT_DIR}")
        return

    # Option 2: Extract 22 contiguous axial slices spanning the adrenal tumor from the NIfTI volume
    if os.path.isfile(NIFTI_PATH):
        import nibabel as nib

        vol = nib.load(NIFTI_PATH).get_fdata()
        # Slices 54..75 (22 slices) span the full adrenal tumor for Adrenal_Ki67_Seg_041 (peak at index 14 -> slice 68)
        start_slice = 54
        num_slices = 22
        for i in range(num_slices):
            slice_idx = start_slice + i
            hu = vol[:, :, slice_idx].astype(np.float32)
            windowed = np.clip(hu, -160, 240)
            normalized = ((windowed + 160) / 400.0 * 255.0).astype(np.uint8)
            out_path = os.path.join(OUTPUT_DIR, f"slice_{i:03d}.png")
            cv2.imwrite(out_path, normalized)
        print(f"Saved {num_slices} PNG slices to {OUTPUT_DIR}")
        return

    raise FileNotFoundError("Neither DICOM_DIR nor NIFTI_PATH found for Adrenal_Ki67_Seg_041.")


if __name__ == "__main__":
    convert_patient_041()
