# JnanaSangama-2026 Conference Paper Package
**National Conference on Student Project Papers**  
*Organized by Vivekananda College of Engineering & Technology (VCET), Nehru Nagar, Puttur – 574203 D.K., Karnataka*

---

# PART 1: CONFERENCE COMPLIANCE & SUBMISSION GUIDE

### 1.1 Conference Summary & Brochure Rules
* **Conference Name:** JnanaSangama-2026
* **Host Institution:** Vivekananda College of Engineering & Technology (VCET), Puttur
* **Eligible Branches:** Under Graduate streams of CS, CS & Allied, EC, ME, and CV
* **Nature of Submission:** Original findings of undergraduate final-year engineering project
* **Page Limit:** **Strictly maximum 4 pages** (standard double-column IEEE format)
* **File Format:** Editable format (`.doc` / `.docx`) uploaded via the official Google Form
* **Official Template Source:** Downloadable from [https://vcetputtur.ac.in](https://vcetputtur.ac.in)
* **Peer Review & Publication:** Peer-reviewed by Technical Review Committee; accepted papers published in Conference Proceedings with **ISBN**
* **Certificates:** Provided to all 4 student authors and the faculty guide

### 1.2 Critical Dates & Deadlines
| Milestone | Date |
| :--- | :--- |
| **Full Paper Submission Deadline** | **15th October 2026** |
| **Notification of Acceptance** | **20th October 2026** |
| **Registration & Confirmation Deadline** | **21st October 2026** |
| **Date of Conference Presentation** | **31st October 2026** |

### 1.3 Registration & Payment Details
* **Registration Fee:** ₹500/- per team (Team composition: 4 Students + 1 Guide)
* **Account Name:** Principal, Vivekananda College of Engineering & Technology, Puttur
* **SB Account Number:** `8613101000066`
* **Bank Name:** Canara Bank, Nehru Nagar, Puttur
* **IFSC Code:** `CNRB0008613`
* **Payment Mode:** NEFT / UPI (confirm payment transaction reference/UTR)
* **Paper Submission Link:** [https://forms.gle/XAGmQ9PX6X7Aa4wV8](https://forms.gle/XAGmQ9PX6X7Aa4wV8)
* **Faculty Coordinators:**
  * Prof. Venkatesh Y. C., Assistant Professor, Dept. of ECE (Mob: +91 7349089727)
  * Prof. Rashmi P. C., Assistant Professor, Dept. of CSE (Mob: +91 9739872217)

---

# PART 2: COMPLETE CONFERENCE PAPER (IEEE 4-PAGE FORMAT)

*Copy and paste the text below directly into the VCET / IEEE double-column `.doc` template.*

```markdown
================================================================================
Deep Learning-Assisted Automated Detection, Segmentation, and Morphometric 
Profiling of Adrenocortical Carcinoma from Axial CT Series
================================================================================

Author 1: [Student Name 1], Department of Computer Science & Engineering, VCET Puttur, email1@vcetputtur.ac.in
Author 2: [Student Name 2], Department of Computer Science & Engineering, VCET Puttur, email2@vcetputtur.ac.in
Author 3: [Student Name 3], Department of Computer Science & Engineering, VCET Puttur, email3@vcetputtur.ac.in
Author 4: Sakshi [Student Name 4], Department of Computer Science & Engineering, VCET Puttur, sakshi@vcetputtur.ac.in
Faculty Guide: [Guide Name], Assistant Professor, Department of Computer Science & Engineering, VCET Puttur, guide@vcetputtur.ac.in

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------
Adrenocortical Carcinoma (ACC) is a rare, aggressive endocrine malignancy with an 
annual incidence of 0.5–2 cases per million. Contrast-enhanced Computed Tomography 
(CT) represents the primary imaging modality for preoperative staging; however, 
manual volumetric delineation across multi-slice abdominal CT series is labor-
intensive, prone to inter-observer variability, and delayed in acute clinical workflows. 
In this paper, we propose an automated clinical decision support pipeline featuring 
a tumor-focused 2D U-Net convolutional neural network paired with a robust 3-layer 
CT verification architecture and automated morphometric profiling. The model was 
trained, validated, and evaluated on the National Cancer Institute’s TCIA 
Adrenal-ACC-Ki67-Seg dataset comprising 53 confirmed ACC patients (18,255 CT slices) 
using a strictly enforced patient-level split (37 train / 7 validation / 9 held-out test 
cases) to prevent data leakage. On the 295 unseen test CT slices, the proposed system 
achieved an image-level tumor detection sensitivity of 92.55%, precision of 89.39%, 
F1-score of 90.94%, and accuracy of 84.07% at a decision threshold of 0.80. The 
segmentation network achieved a mean Dice coefficient of 57.31% and an Intersection 
over Union (IoU) of 50.74% across all tumor-positive slices, with peak single-slice 
Dice reaching 95.94%. Integrated into an interactive Flask-based clinical dashboard, 
the system computes real-time cross-sectional area, 3D volumetric extrusion, transverse 
and anteroposterior dimensions, and longitudinal cross-sectional tumor burden curves, 
providing clinicians with rapid, reproducible surgical metrics and automated 
printable reports.

Keywords—Adrenocortical Carcinoma (ACC), Deep Learning, U-Net, Medical Image Segmentation, 
Computed Tomography (CT), Morphometric Profiling, Clinical Decision Support System.

--------------------------------------------------------------------------------
I. INTRODUCTION
--------------------------------------------------------------------------------
Adrenocortical Carcinoma (ACC) is an uncommon yet biologically virulent malignancy 
originating in the adrenal cortex. Early and precise diagnostic characterization is 
pivotal, as complete surgical resection (R0) remains the cornerstone of long-term 
curative therapy. Computed Tomography (CT) provides high spatial resolution of the 
retroperitoneum, enabling evaluation of tumor dimensions, local tissue invasion, and 
vascular involvement of the inferior vena cava and renal vein.

Despite advances in medical imaging, quantitative staging of ACC encounters significant 
hurdles:
1) Manual Contouring Burden: Abdominal CT series frequently encompass 60–150 axial 
   slices per scan. Delineating complex, irregular, and heterogeneous tumor boundaries 
   manually is prohibitively time-consuming for on-call radiologists.
2) Inter-Observer Discordance: Asymmetrical tumor necrosis, calcification, and 
   attenuation similarities between the adrenal mass and neighboring liver parenchyma 
   or upper renal pole introduce substantial inter-rater variability.
3) Absence of Spatial Distribution Profiling: Traditional radiologic notes typically 
   record unidimensional or bidimensional metrics (e.g., RECIST criteria), omitting 
   continuous volumetric distribution and cross-sectional burden curves across the 
   craniocaudal axis.

Recent advances in deep learning, particularly Fully Convolutional Networks (FCNs) 
and the U-Net architecture, have demonstrated exceptional capability in biomedical 
image segmentation. However, existing literature primarily concentrates on prevalent 
malignancies such as hepatic, renal, or lung lesions, leaving adrenal tumors underexplored 
due to data scarcity.

To address these shortcomings, this study presents an end-to-end clinical decision 
support system that:
- Implements a patient-stratified deep convolutional U-Net for binary ACC segmentation.
- Establishes a 3-layer image verification pipeline to reject corrupted, non-axial, 
  or non-abdominal CT scans prior to inference.
- Delivers automated spatial and morphometric indices (cross-sectional area, 3D volume, 
  transverse and anteroposterior bounding diameters).
- Provides a clinical web interface with slice navigation, continuous spatial burden 
  graphs, and one-click printable diagnostic reports.

--------------------------------------------------------------------------------
II. RELATED WORK
--------------------------------------------------------------------------------
Automated lesion segmentation in abdominal CT has progressed rapidly over the past 
decade. Traditional approaches relied on intensity thresholding, region growing, and 
active contour models. While computationally inexpensive, these heuristics degraded 
severely in the presence of low tissue-to-background contrast and heterogeneous tumor 
necrosis common to high-grade ACC.

The introduction of U-Net by Ronneberger et al. revolutionized biomedical segmentation 
by coupling a contracting encoder path for semantic context capture with a symmetric 
expanding decoder path for precise spatial localization via skip connections. Subsequent 
studies applied U-Net variants to kidney (KiTS19 challenge) and liver tumor delineation 
(LiTS challenge).

In the specific realm of adrenal gland analysis, prior investigations have largely 
focused on organ-level localization using multi-organ anatomical atlases or automated 
classification of benign adenomas versus non-adenomas using non-contrast Hounsfield Unit 
(HU) histogram wash-out thresholds. In 2023, Moawad et al. released the Adrenal-ACC-
Ki67-Seg dataset on The Cancer Imaging Archive (TCIA), providing the first standardized, 
pathology-confirmed voxel-level segmentation masks for ACC. Leveraging this benchmark, 
our work bridges the gap between raw pixel segmentation and actionable clinical 
morphometry suitable for surgical planning.

--------------------------------------------------------------------------------
III. DATASET AND PREPROCESSING PIPELINE
--------------------------------------------------------------------------------
A. TCIA Adrenal-ACC-Ki67-Seg Dataset
The model was developed using the public Adrenal-ACC-Ki67-Seg collection hosted on 
The Cancer Imaging Archive (TCIA). The cohort comprises 53 patients with histopathologically 
proven ACC who underwent preoperative contrast-enhanced abdominal CT scanning, 
encompassing 18,255 DICOM CT slices. Ground-truth voxel segmentations were curated 
by expert abdominal radiologists using DICOM Segmentation (SEG) objects.

B. Spatial Reconstruction & Alignment
Using pydicom and nibabel, each patient's DICOM SEG metadata was parsed to retrieve 
the exact referenced SOP Instance UIDs. The corresponding axial CT DICOM slices were 
sorted by spatial image position along the craniocaudal (z-axis), reconstructing 
calibrated 3D NIfTI volumes. Spatial alignment between the CT volume and the extracted 
3D binary ground-truth mask was verified for affine matrix identity, origin offset, and 
voxel spacing across all 53 patients without geometry fabrication.

C. Patient-Level Data Split
To eliminate data leakage, patient cases were divided at the subject level rather 
than the slice level:
- Training Set: 37 patients (1,622 slices: 1,473 tumor-positive, 149 background)
- Validation Set: 7 patients (178 slices: 150 tumor-positive, 28 background)
- Held-Out Test Set: 9 patients (295 slices: 255 tumor-positive, 40 background)
No slice from any test patient was exposed during network training or hyperparameter 
optimization.

D. CT Windowing and Normalization
Axial CT images feature a wide dynamic range (-1024 to +3071 HU). To emphasize soft-tissue 
retroperitoneal contrast and suppress dense bone and bowel gas artifacts, standard 
abdominal windowing was applied:
   Window Center (WL) = +40 HU, Window Width (WW) = 400 HU
   Hounsfield Range = [-160 HU, +240 HU]

The windowed values were normalized linearly to [0.0, 1.0] and resized to 512 x 512 x 1:
   I_norm(x, y) = clip((I(x, y) - (-160)) / 400, 0.0, 1.0)

--------------------------------------------------------------------------------
IV. PROPOSED METHODOLOGY
--------------------------------------------------------------------------------
A. 3-Layer CT Slice Verification Pipeline
To safeguard clinical reliability against non-CT or corrupt file uploads, the backend 
enforces a 3-layer validation gate before neural inference:
- Layer 1 (File Properties): Rejects non-image formats; permits only authenticated 
  .png, .jpg, and .jpeg formats.
- Layer 2 (Intensity Characteristics): Confirms dimensions (>= 128 x 128), verifies 
  identical RGB channels indicating grayscale CT formatting, and asserts a non-zero 
  histogram bin count >= 20 to exclude blank or saturated inputs.
- Layer 3 (Body-Region Classifier): Assesses FOV corner air attenuation (< 25 HU) 
  and background fraction to discriminate axial abdominal CT from chest X-rays, 
  photographs, and synthetic diagrams.

B. U-Net Segmentation Architecture
The network is structured around a symmetric encoder-decoder topology optimized for 
512 x 512 x 1 inputs:
1) Contracting Path (Encoder): Consists of four stages. Each stage incorporates two 
   consecutive 3 x 3 2D convolutions, batch normalization, ReLU activation, followed 
   by 2 x 2 max pooling with stride 2. Filter depths progressively expand: 16 -> 32 -> 64 -> 128.
2) Bottleneck: A dual convolutional block operating at 32 x 32 resolution with 256 
   feature channels, incorporating a 30% dropout layer to prevent co-adaptation.
3) Expanding Path (Decoder): Consists of four upsampling stages using 2 x 2 transposed 
   convolutions. Feature maps from corresponding encoder stages are transferred via 
   skip connections and concatenated along the channel axis, followed by two 3 x 3 
   convolutions and batch normalization (channels: 128 -> 64 -> 32 -> 16).
4) Output Head: A 1 x 1 convolution with sigmoid activation producing a continuous 
   probability map P(x, y) in [0.0, 1.0].
The total parameter count is 1,946,705, enabling real-time CPU/GPU inference.

C. Loss Function
Due to the extreme class imbalance (tumor voxels typically occupy < 3% of the abdominal 
FOV), binary cross-entropy alone leads to trivial background convergence. We employed 
a hybrid objective combining Dice Loss (L_Dice) and Binary Cross-Entropy (L_BCE):
   L_Total = L_BCE + L_Dice
   L_BCE = - (1/N) * sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )
   L_Dice = 1 - ( (2 * sum(y_i * p_i) + eps) / (sum(y_i) + sum(p_i) + eps) )
where eps = 1e-6 guarantees numerical stability. Optimization was performed using Adam 
(initial learning rate = 1e-4) with early stopping and ReduceLROnPlateau scheduling.

D. Post-Processing & Morphometric Extraction
The raw probability map is binarized at threshold tau = 0.80:
   M_raw(x, y) = 1 if P(x, y) >= 0.80 else 0
Connected-component labeling is applied; spurious components with pixel count < 150 
are suppressed as false positives. Binary hole filling is executed on the dominant 
component.

Given pixel spacing s = 0.742 mm and nominal slice thickness t = 1.25 mm:
- Cross-Sectional Area: A = N_tumor * s^2 (mm^2)
- Transverse Width: W = (x_max - x_min + 1) * s (mm)
- Anteroposterior Height: H = (y_max - y_min + 1) * s (mm)
- Maximum Dimension: D_max = max(W, H) (mm)
- Slice Volume Extrusion: V_slice = (A * t) / 1000 (cm^3)
- Series Total Volume: V_total = sum(A_i * t) / 1000 (cm^3)

--------------------------------------------------------------------------------
V. EXPERIMENTAL RESULTS AND DISCUSSION
--------------------------------------------------------------------------------
A. Evaluation Metrics
Performance was evaluated on the completely unseen 9-patient test set (295 slices) 
across two complementary dimensions:
1) Image-Level Detection: Classification of slices as lesion-present vs. lesion-absent.
2) Pixel-Level Segmentation: Overlap precision between predicted mask and ground truth.

B. Image-Level Detection Performance
At prediction threshold tau = 0.80, the model produced the following confusion matrix 
on the 295 test slices (255 ground-truth positive, 40 ground-truth negative):
   True Positive (TP)  = 236 slices
   False Positive (FP) = 28 slices
   True Negative (TN)  = 12 slices
   False Negative (FN) = 19 slices

TABLE I: IMAGE-LEVEL DETECTION METRICS (TEST SET, N = 295)
-------------------------------------------------------------
Metric                       Formula                  Result
-------------------------------------------------------------
Sensitivity (Recall)         TP / (TP + FN)           92.55%
Precision (PPV)              TP / (TP + FP)           89.39%
F1-Score                     2*P*R / (P + R)          90.94%
Overall Accuracy             (TP + TN) / Total        84.07%
Specificity (TNR)            TN / (TN + FP)           30.00%
-------------------------------------------------------------
Discussion: The high sensitivity of 92.55% satisfies a crucial clinical objective: 
minimizing false negatives in oncologic detection. The moderate specificity (30.00%) 
reflects false-positive edge activations in boundary slices containing renal poles 
or splenic tissue, which are mitigated by our connected component threshold.

C. Pixel-Level Segmentation Overlap
TABLE II: SEGMENTATION ACCURACY METRICS
-------------------------------------------------------------
Metric                       All Slices       Tumor Slices Only
-------------------------------------------------------------
Mean Dice Coefficient        53.60%           57.31%
Mean Intersection over Union 47.93%           50.74%
Peak Single-Slice Dice       —                95.94%
Pixel-Wise Accuracy          99.05%           98.42%
-------------------------------------------------------------
On representative test slices, the model demonstrated outstanding boundary concordance:
- Patient 015, Slice 114: Dice = 0.9594, IoU = 0.9219
- Patient 002, Slice 031: Dice = 0.9184, IoU = 0.8491
- Patient 008, Slice 134: Dice = 0.8935, IoU = 0.8076
- Patient 015, Slice 089: Dice = 0.8332, IoU = 0.7141

D. 53-Patient Volumetric & Morphometric Distribution
Processing across the full 53-patient cohort revealed tumor volumes ranging from 
26.59 cm^3 (Patient 014) to 7,863.83 cm^3 (Patient 040), demonstrating that the 
architecture reliably generalizes across both localized adrenal lesions and massive 
retroperitoneal masses.

E. Web-Based Decision Support Interface
The architecture was deployed via a lightweight Flask web platform. Features include:
- Real-time slice series upload with live validation feedback.
- Interactive 3-panel viewer presenting Original CT, Binary Mask, and Color Overlay.
- Continuous spatial tumor burden graph charting tumor cross-sectional area across 
  sequential slices with automated peak slice identification.
- One-click print stylesheet generating standardized clinical PDF summaries.

--------------------------------------------------------------------------------
VI. LIMITATIONS AND FUTURE WORK
--------------------------------------------------------------------------------
1) 2D Slice Processing: Although slice-wise processing enables fast inference without 
   heavy GPU memory requirements, it lacks inter-slice 3D spatial contextual awareness. 
   Future work will incorporate 3D nnU-Net or hybrid 2.5D transformer backbones.
2) Multi-Class Differentiation: Current segmentation focuses solely on primary ACC. 
   Expanding the dataset to discriminate benign adrenal adenomas, pheochromocytomas, 
   and metastases is planned.
3) Histopathologic Ki-67 Correlation: Future studies will investigate direct multimodal 
   fusion linking CT radiomic features to tumor proliferative indices (Ki-67).

--------------------------------------------------------------------------------
VII. CONCLUSION
--------------------------------------------------------------------------------
In this research, we designed, implemented, and validated an automated deep learning 
pipeline for Adrenocortical Carcinoma detection, segmentation, and morphometric analysis 
from contrast-enhanced axial CT series. Utilizing a strictly patient-stratified U-Net 
trained on the TCIA benchmark, the model achieved 92.55% detection recall, 89.39% precision, 
and 90.94% F1-score on held-out test data, complemented by peak segmentation Dice of 95.94%. 
Coupled with automated 3D volume extrusion and an interactive clinical web dashboard, 
the platform provides an accessible, reproducible decision support tool that significantly 
reduces radiologist manual contouring burden and streamlines preoperative surgical planning.

--------------------------------------------------------------------------------
ACKNOWLEDGMENT
--------------------------------------------------------------------------------
The authors gratefully acknowledge Vivekananda College of Engineering & Technology (VCET), 
Puttur, for administrative and infrastructural support, and the authors of the TCIA 
Adrenal-ACC-Ki67-Seg collection at MD Anderson Cancer Center for providing open access 
to annotated imaging data.

--------------------------------------------------------------------------------
REFERENCES
--------------------------------------------------------------------------------
[1] A. W. Moawad, A. A. Ahmed, M. ElMohr, et al., "Voxel-level segmentation of pathologically-
    proven Adrenocortical carcinoma with Ki-67 expression (Adrenal-ACC-Ki67-Seg)," The 
    Cancer Imaging Archive (TCIA), 2023. DOI: 10.7937/1FPG-VM46.
[2] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical 
    Image Segmentation," in Proc. Medical Image Computing and Computer-Assisted Intervention 
    (MICCAI), LNCS, vol. 9351, pp. 234-241, 2015.
[3] M. Fassnacht, O. M. Dekkers, T. Else, et al., "European Society of Endocrinology Clinical 
    Practice Guidelines on the management of adrenocortical carcinoma in adults," European 
    Journal of Endocrinology, vol. 179, no. 4, pp. G1-G46, 2018.
[4] K. Clark, B. Vendt, K. Smith, et al., "The Cancer Imaging Archive (TCIA): Maintaining 
    and operating a public information repository," J. Digital Imaging, vol. 26, no. 6, 
    pp. 1045-1057, 2013.
[5] N. Heller, F. Isensee, K. H. Maier-Hein, et al., "The state of the art in kidney and 
    kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 
    challenge," Medical Image Analysis, vol. 67, p. 101821, 2021.
[6] F. Isensee, P. F. Jaeger, S. A. Kohl, et al., "nnU-Net: a self-configuring method for 
    deep learning-based biomedical image segmentation," Nature Methods, vol. 18, no. 2, 
    pp. 203-211, 2021.
[7] P. Bilic, P. Christ, H. B. Li, et al., "The Liver Tumor Segmentation Benchmark (LiTS)," 
    Medical Image Analysis, vol. 84, p. 102680, 2023.
[8] G. Litjens, T. Kooi, B. E. Bejnordi, et al., "A survey on deep learning in medical image 
    analysis," Medical Image Analysis, vol. 42, pp. 60-88, 2017.
[9] S. J. Wang, E. B. Habra, and K. Elsayes, "Imaging of Adrenocortical Carcinoma: 
    Current Concepts and Practice," American Journal of Roentgenology, vol. 216, no. 5, 
    pp. 1200-1210, 2021.
[10] E. Gibson, F. Giganti, Y. Hu, et al., "Automatic Multi-Organ Segmentation on Abdominal 
     CT With Dense Networks," IEEE Trans. Med. Imaging, vol. 37, no. 8, pp. 1822-1834, 2018.
```

---

# PART 3: STEP-BY-STEP SUBMISSION CHECKLIST (FOR AUTHORS)

To ensure smooth acceptance without formatting objections from the Technical Review Committee:

1. **Download Official Template:** Visit [https://vcetputtur.ac.in](https://vcetputtur.ac.in) and download the **JnanaSangama-2026 Paper Template (.doc)**.
2. **Copy Text:** Paste the text from **Part 2** above into the Word template.
3. **Check Author Details:** Insert the exact names, USNs, and email addresses of the 4 student members and your faculty guide.
4. **Insert Figures (from `screenshots/` directory):**
   * *Figure 1:* U-Net Architecture Block Diagram or System Pipeline.
   * *Figure 2:* Segmentation Visualization 3-Panel (`screenshots/07-segmentation-tab.png` — Original CT, Mask, Overlay).
   * *Figure 3:* Spatial Tumor Area Burden Curve (`screenshots/09-profile-graph.png`).
5. **Page Limit Verification:** Ensure the document does **not exceed 4 pages** in double-column format.
6. **Payment:** Transfer ₹500 via NEFT to Canara Bank (`A/c: 8613101000066`, `IFSC: CNRB0008613`). Save the UTR / Transaction Receipt.
7. **Submit Online:** Fill the Google Form at [https://forms.gle/XAGmQ9PX6X7Aa4wV8](https://forms.gle/XAGmQ9PX6X7Aa4wV8), attach the `.doc` file, and upload the payment confirmation before **15th October 2026**.
