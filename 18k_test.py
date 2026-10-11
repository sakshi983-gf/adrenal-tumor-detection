from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(
    r"D:\2026\Tumor Detection\Code\001_Adrenal_Tumor_Detection_System"
)

TEST_IMAGE_DIR = PROJECT_ROOT / "dataset" / "unet_dataset" / "test" / "images"
TEST_MASK_DIR = PROJECT_ROOT / "dataset" / "unet_dataset" / "test" / "masks"

print("=" * 60)
print("STEP 18K-TEST")
print("Finding tumor-free/background slices")
print("=" * 60)

found = 0

for mask_path in sorted(TEST_MASK_DIR.glob("*.png")):

    mask = np.array(
        Image.open(mask_path).convert("L")
    )

    tumor_pixels = np.count_nonzero(mask)

    if tumor_pixels == 0:

        image_path = TEST_IMAGE_DIR / mask_path.name

        if image_path.exists():

            print()
            print("Background test slice found")
            print("-" * 40)
            print("Patient/Slice :", mask_path.stem)
            print("Image         :", image_path)
            print("Mask          :", mask_path)
            print("Image size    :", Image.open(image_path).size)
            print("Tumor pixels  :", tumor_pixels)

            found += 1

            if found == 5:
                break

print()
print("=" * 60)
print("Total background slices displayed:", found)
print("=" * 60)