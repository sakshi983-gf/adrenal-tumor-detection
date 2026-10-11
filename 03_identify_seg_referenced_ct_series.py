from pathlib import Path
from collections import defaultdict
import json
import pydicom


# ============================================================
# STEP 03
# Identify the exact CT series referenced by each DICOM SEG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
TCIA_ROOT = PROJECT_ROOT / "raw" / "adrenal_acc_ki67_seg"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "metadata" / "step03"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "seg_referenced_ct_series.json"


def is_patient_folder(path):
    return path.is_dir() and path.name.startswith("Adrenal_Ki67_Seg_")


def read_dicom(path):
    try:
        return pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=False
        )
    except Exception:
        return None


def get_referenced_ct_uids(seg_ds):
    """
    Extract CT SOP Instance UIDs referenced by the DICOM SEG.
    """

    referenced_uids = set()

    if not hasattr(seg_ds, "ReferencedSeriesSequence"):
        return referenced_uids

    for series_item in seg_ds.ReferencedSeriesSequence:

        if not hasattr(series_item, "ReferencedInstanceSequence"):
            continue

        for instance in series_item.ReferencedInstanceSequence:

            if hasattr(instance, "ReferencedSOPInstanceUID"):
                referenced_uids.add(
                    str(instance.ReferencedSOPInstanceUID)
                )

    return referenced_uids


def main():

    print("=" * 70)
    print("STEP 03 - SEG REFERENCED CT SERIES IDENTIFICATION")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nTCIA dataset:")
    print(TCIA_ROOT)

    if not TCIA_ROOT.exists():
        print("\nERROR: TCIA dataset folder not found.")
        return

    patient_folders = sorted(
        [
            p for p in TCIA_ROOT.iterdir()
            if is_patient_folder(p)
        ],
        key=lambda p: p.name
    )

    print(f"\nPatients found: {len(patient_folders)}")

    reports = []

    total_seg = 0
    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process every patient
    # --------------------------------------------------------

    for patient_index, patient_dir in enumerate(patient_folders, start=1):

        patient_id = patient_dir.name

        print("\n" + "-" * 70)
        print(
            f"[{patient_index}/{len(patient_folders)}] "
            f"{patient_id}"
        )
        print("-" * 70)

        dicom_files = []

        for path in patient_dir.rglob("*"):

            if not path.is_file():
                continue

            ds = read_dicom(path)

            if ds is not None:
                dicom_files.append((path, ds))

        print(f"Readable DICOM objects: {len(dicom_files)}")

        # ----------------------------------------------------
        # Find SEG objects
        # ----------------------------------------------------

        seg_objects = []

        for path, ds in dicom_files:

            modality = str(
                getattr(ds, "Modality", "")
            ).upper()

            if modality == "SEG":
                seg_objects.append((path, ds))

        print(f"SEG objects: {len(seg_objects)}")

        if len(seg_objects) == 0:

            failed += 1

            reports.append({
                "patient_id": patient_id,
                "status": "NO_SEG_FOUND"
            })

            continue

        total_seg += len(seg_objects)

        # ----------------------------------------------------
        # Use each SEG
        # ----------------------------------------------------

        for seg_path, seg_ds in seg_objects:

            print(f"\nSEG file:")
            print(seg_path)

            # ------------------------------------------------
            # Segment information
            # ------------------------------------------------

            segments = []

            if hasattr(seg_ds, "SegmentSequence"):

                for segment in seg_ds.SegmentSequence:

                    segment_number = str(
                        getattr(segment, "SegmentNumber", "")
                    )

                    label = str(
                        getattr(segment, "SegmentLabel", "")
                    )

                    description = str(
                        getattr(
                            segment,
                            "SegmentDescription",
                            ""
                        )
                    )

                    segments.append({
                        "segment_number": segment_number,
                        "label": label,
                        "description": description
                    })

                    print(
                        f"Segment {segment_number}: "
                        f"{label} | {description}"
                    )

            # ------------------------------------------------
            # Extract referenced CT SOP Instance UIDs
            # ------------------------------------------------

            referenced_uids = get_referenced_ct_uids(seg_ds)

            print(
                f"Referenced CT SOP Instance UIDs: "
                f"{len(referenced_uids)}"
            )

            # ------------------------------------------------
            # Build CT lookup
            # ------------------------------------------------

            ct_lookup = {}

            for path, ds in dicom_files:

                modality = str(
                    getattr(ds, "Modality", "")
                ).upper()

                if modality != "CT":
                    continue

                sop_uid = str(
                    getattr(ds, "SOPInstanceUID", "")
                )

                if sop_uid:
                    ct_lookup[sop_uid] = {
                        "path": str(path),
                        "series_uid": str(
                            getattr(
                                ds,
                                "SeriesInstanceUID",
                                ""
                            )
                        ),
                        "series_description": str(
                            getattr(
                                ds,
                                "SeriesDescription",
                                ""
                            )
                        ),
                        "instance_number": str(
                            getattr(
                                ds,
                                "InstanceNumber",
                                ""
                            )
                        )
                    }

            # ------------------------------------------------
            # Match referenced UIDs to CT files
            # ------------------------------------------------

            located = []
            missing = []

            series_groups = defaultdict(list)

            for uid in sorted(referenced_uids):

                if uid in ct_lookup:

                    info = ct_lookup[uid]

                    located.append({
                        "sop_instance_uid": uid,
                        **info
                    })

                    series_groups[
                        info["series_uid"]
                    ].append(uid)

                else:

                    missing.append(uid)

            print(
                f"Referenced CT files located: "
                f"{len(located)}"
            )

            print(
                f"Referenced CT files missing: "
                f"{len(missing)}"
            )

            # ------------------------------------------------
            # Identify referenced CT series
            # ------------------------------------------------

            referenced_series = []

            for series_uid, uids in sorted(
                series_groups.items()
            ):

                series_info = {
                    "series_instance_uid": series_uid,
                    "referenced_slice_count": len(uids),
                    "series_description": ""
                }

                descriptions = set()

                for uid in uids:

                    description = ct_lookup[uid][
                        "series_description"
                    ]

                    if description:
                        descriptions.add(description)

                if descriptions:
                    series_info["series_description"] = (
                        sorted(descriptions)[0]
                    )

                referenced_series.append(series_info)

            print(
                f"Referenced CT series identified: "
                f"{len(referenced_series)}"
            )

            for series in referenced_series:

                print(
                    f"  Series UID: "
                    f"{series['series_instance_uid']}"
                )

                print(
                    f"  Referenced slices: "
                    f"{series['referenced_slice_count']}"
                )

                print(
                    f"  Description: "
                    f"{series['series_description']}"
                )

            # ------------------------------------------------
            # Determine status
            # ------------------------------------------------

            if (
                len(referenced_series) == 1
                and len(missing) == 0
            ):

                status = (
                    "SEG_REFERENCED_CT_SERIES_IDENTIFIED"
                )

                successful += 1

            elif len(referenced_series) > 0:

                status = (
                    "SEG_REFERENCED_CT_SERIES_PARTIALLY_IDENTIFIED"
                )

                failed += 1

            else:

                status = (
                    "SEG_REFERENCED_CT_SERIES_NOT_IDENTIFIED"
                )

                failed += 1

            print(f"\nStatus: {status}")

            reports.append({
                "patient_id": patient_id,
                "seg_file": str(seg_path),
                "segments": segments,
                "referenced_ct_sop_instance_count": len(
                    referenced_uids
                ),
                "located_ct_count": len(located),
                "missing_ct_count": len(missing),
                "missing_ct_sop_instance_uids": missing,
                "referenced_ct_series": referenced_series,
                "status": status
            })

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    final_report = {
        "project": "AI-Based Adrenal Tumor Detection and Segmentation System",
        "step": "03",
        "description": (
            "Identification of CT series referenced "
            "by original TCIA DICOM SEG objects"
        ),
        "patients_inspected": len(patient_folders),
        "total_seg_objects": total_seg,
        "successful_patients": successful,
        "failed_patients": failed,
        "reports": reports
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_report,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("STEP 03 COMPLETE")
    print("=" * 70)

    print(f"\nPatients inspected: {len(patient_folders)}")
    print(f"SEG objects: {total_seg}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    print("\nReport:")
    print(REPORT_FILE)

    if (
        len(patient_folders) == 53
        and total_seg == 53
        and failed == 0
    ):

        print(
            "\nSTEP 03 STATUS: "
            "ALL SEG-REFERENCED CT SERIES IDENTIFIED"
        )

    else:

        print(
            "\nSTEP 03 STATUS: "
            "REVIEW REQUIRED"
        )


if __name__ == "__main__":
    main()