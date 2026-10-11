/**
 * AdrenalCare — Protected Clinical Dashboard Controller
 * Handles:
 *  - Single slice & slice series upload (no minimum slice count)
 *  - 3-Layer CT Validation error display
 *  - 4-Tab navigation (Overview, Segmentation, Measurements, Profile)
 *  - Slice navigation & Chart.js cross-sectional area profile rendering (1 or N slices)
 *  - Printable clinical report generation (window.print)
 *  - Header user dropdown (/profile, /settings, /logout)
 */

(function () {
    "use strict";

    // =========================================================================
    // STATE
    // =========================================================================
    let selectedFiles = [];
    let currentSeriesData = null;
    let activeSliceIndex = 0;
    let profileChartInstance = null;

    // =========================================================================
    // DOM ELEMENTS
    // =========================================================================
    const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
    const appSidebar = document.getElementById("app-sidebar");
    const sidebarBackdrop = document.getElementById("sidebar-backdrop");

    // User Dropdown & Header Actions
    const generateReportBtn = document.getElementById("generate-report-btn");
    const headerUserBtn = document.getElementById("header-user-btn");
    const headerUserDropdown = document.getElementById("header-user-dropdown");

    // Upload Panel
    const dropZone = document.getElementById("drop-zone");
    const selectSingleBtn = document.getElementById("select-single");
    const selectSeriesBtn = document.getElementById("select-series");
    const singleInput = document.getElementById("file-single");
    const seriesInput = document.getElementById("file-series");
    const statusLabel = document.getElementById("file-status");
    const runBtn = document.getElementById("run-btn");

    // Patient Info
    const patientIdInput = document.getElementById("patient-id");
    const patientAgeInput = document.getElementById("patient-age");
    const patientSexInput = document.getElementById("patient-sex");

    // Sidebar Findings
    const sbTumorLocation = document.getElementById("sb-tumor-location");
    const sbEstimatedSize = document.getElementById("sb-estimated-size");
    const sbEstimatedVolume = document.getElementById("sb-estimated-volume");
    const sbDetectionConfidence = document.getElementById("sb-detection-confidence");

    // Sidebar Parameters & Settings
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdValue = document.getElementById("threshold-value");
    const lastAnalyzed = document.getElementById("last-analyzed");
    const runNewAnalysisBtn = document.getElementById("run-new-analysis-btn");

    // Tabs & Main Containers
    const tabButtons = document.querySelectorAll(".dash-tab");
    const errorBox = document.getElementById("error-box");
    const loadingPanel = document.getElementById("loading-panel");
    const loadingTitle = document.getElementById("loading-title");
    const loadingSub = document.getElementById("loading-sub");
    const emptyPanel = document.getElementById("empty-panel");
    const tabPanelsContainer = document.getElementById("tab-panels-container");

    // Tab 1: Overview
    const detectionCard = document.getElementById("detection-card");
    const detHeadline = document.getElementById("det-headline");
    const detConfidence = document.getElementById("det-confidence");
    const detThreshold = document.getElementById("det-threshold");
    const detSubtext = document.getElementById("det-subtext");

    const statSlicesAnalyzed = document.getElementById("stat-slices-analyzed");
    const statPeakArea = document.getElementById("stat-peak-area");
    const statTotalVolume = document.getElementById("stat-total-volume");
    const statMaxDim = document.getElementById("stat-max-dim");
    const statTumorArea = document.getElementById("stat-tumor-area");
    const statConfidence = document.getElementById("stat-confidence");

    // Tab 2: Segmentation
    const prevSliceBtn = document.getElementById("prev-slice-btn");
    const nextSliceBtn = document.getElementById("next-slice-btn");
    const sliceCounter = document.getElementById("slice-counter");
    const jumpPeakBtn = document.getElementById("jump-peak-btn");
    const segOrigImg = document.getElementById("seg-orig-img");
    const segMaskImg = document.getElementById("seg-mask-img");
    const segOverlayImg = document.getElementById("seg-overlay-img");

    // Tab 3: Measurements
    const measSliceBadge = document.getElementById("meas-slice-badge");
    const measArea = document.getElementById("meas-area");
    const measVolume = document.getElementById("meas-volume");
    const measWidth = document.getElementById("meas-width");
    const measHeight = document.getElementById("meas-height");
    const measMaxDim = document.getElementById("meas-max-dim");
    const measCoverage = document.getElementById("meas-coverage");

    // Tab 4: Profile
    const profileContent = document.getElementById("profile-content");

    // =========================================================================
    // HEADER DROPDOWN & GENERATE REPORT BUTTON
    // =========================================================================
    if (generateReportBtn) {
        generateReportBtn.addEventListener("click", function () {
            window.print();
        });
    }

    if (headerUserBtn && headerUserDropdown) {
        headerUserBtn.addEventListener("click", function (e) {
            e.stopPropagation();
            const isHidden = headerUserDropdown.hidden;
            headerUserDropdown.hidden = !isHidden;
            headerUserBtn.setAttribute("aria-expanded", String(isHidden));
        });

        document.addEventListener("click", function () {
            headerUserDropdown.hidden = true;
            headerUserBtn.setAttribute("aria-expanded", "false");
        });
    }

    // =========================================================================
    // RESPONSIVE SIDEBAR TOGGLE
    // =========================================================================
    if (sidebarToggleBtn && appSidebar) {
        sidebarToggleBtn.addEventListener("click", function () {
            if (window.innerWidth < 1024) {
                const open = appSidebar.classList.toggle("drawer-open");
                if (sidebarBackdrop) sidebarBackdrop.hidden = !open;
            } else {
                appSidebar.classList.toggle("sidebar-collapsed");
            }
        });
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener("click", function () {
            appSidebar.classList.remove("drawer-open");
            sidebarBackdrop.hidden = true;
        });
    }

    // =========================================================================
    // 4-TAB NAVIGATION (Overview, Segmentation, Measurements, Profile)
    // =========================================================================
    function switchTab(tabName) {
        tabButtons.forEach(function (btn) {
            const isMatch = btn.getAttribute("data-tab") === tabName;
            btn.classList.toggle("active", isMatch);
            btn.setAttribute("aria-selected", String(isMatch));
        });

        const panels = {
            overview: document.getElementById("tab-overview"),
            segmentation: document.getElementById("tab-segmentation"),
            measurements: document.getElementById("tab-measurements"),
            profile: document.getElementById("tab-profile")
        };

        Object.keys(panels).forEach(function (key) {
            if (panels[key]) {
                panels[key].hidden = key !== tabName;
                panels[key].classList.toggle("active", key === tabName);
            }
        });

        if (tabName === "profile" && profileChartInstance) {
            setTimeout(function () {
                profileChartInstance.resize();
            }, 20);
        }
    }

    tabButtons.forEach(function (btn) {
        btn.addEventListener("click", function () {
            switchTab(this.getAttribute("data-tab"));
        });
    });

    // =========================================================================
    // UPLOAD PANEL (Single Slice + Slice Series + Drag & Drop)
    // =========================================================================
    function handleFiles(files) {
        const arr = Array.from(files || []).filter(function (f) {
            const name = (f.name || "").toLowerCase();
            return name && !name.startsWith(".") && name !== "thumbs.db" && name !== "desktop.ini";
        });

        arr.sort(function (a, b) {
            return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" });
        });

        selectedFiles = arr;
        const count = selectedFiles.length;

        if (count === 0) {
            statusLabel.textContent = "No files selected";
            runBtn.textContent = "Run Analysis";
            runBtn.disabled = true;
            return;
        }

        statusLabel.textContent = count === 1
            ? "1 slice selected"
            : `${count} slices selected`;
        runBtn.textContent = "Run Analysis";
        runBtn.disabled = false;
    }

    if (selectSingleBtn) {
        selectSingleBtn.addEventListener("click", function () {
            singleInput.click();
        });
    }

    if (selectSeriesBtn) {
        selectSeriesBtn.addEventListener("click", function () {
            seriesInput.click();
        });
    }

    if (singleInput) {
        singleInput.addEventListener("change", function (e) {
            handleFiles(e.target.files);
        });
    }

    if (seriesInput) {
        seriesInput.addEventListener("change", function (e) {
            handleFiles(e.target.files);
        });
    }

    if (dropZone) {
        dropZone.addEventListener("click", function () {
            singleInput.click();
        });

        dropZone.addEventListener("keydown", function (e) {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                singleInput.click();
            }
        });

        dropZone.addEventListener("dragover", function (e) {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });

        dropZone.addEventListener("dragleave", function () {
            dropZone.classList.remove("dragover");
        });

        dropZone.addEventListener("drop", async function (e) {
            e.preventDefault();
            dropZone.classList.remove("dragover");

            if (!e.dataTransfer) return;

            const items = e.dataTransfer.items;
            if (items && items.length > 0 && typeof items[0].webkitGetAsEntry === "function") {
                const collected = [];
                const readEntriesPromise = function (dirReader) {
                    return new Promise(function (resolve) {
                        dirReader.readEntries(function (entries) {
                            resolve(entries || []);
                        });
                    });
                };

                const traverseEntry = async function (entry) {
                    if (!entry) return;
                    if (entry.isFile) {
                        await new Promise(function (resolve) {
                            entry.file(function (file) {
                                collected.push(file);
                                resolve();
                            });
                        });
                    } else if (entry.isDirectory) {
                        const reader = entry.createReader();
                        let batch = await readEntriesPromise(reader);
                        while (batch.length > 0) {
                            for (let i = 0; i < batch.length; i++) {
                                await traverseEntry(batch[i]);
                            }
                            batch = await readEntriesPromise(reader);
                        }
                    }
                };

                const entries = [];
                for (let i = 0; i < items.length; i++) {
                    const entry = items[i].webkitGetAsEntry();
                    if (entry) entries.push(entry);
                }
                for (let i = 0; i < entries.length; i++) {
                    await traverseEntry(entries[i]);
                }
                if (collected.length > 0) {
                    handleFiles(collected);
                    return;
                }
            }

            if (e.dataTransfer.files) {
                handleFiles(e.dataTransfer.files);
            }
        });
    }

    // =========================================================================
    // ERROR DISPLAY (Clear previous results, keep upload panel visible)
    // =========================================================================
    function showError(message, detail) {
        loadingPanel.hidden = true;
        tabPanelsContainer.hidden = true;
        emptyPanel.hidden = false;
        if (generateReportBtn) {
            generateReportBtn.disabled = true;
        }

        sbTumorLocation.textContent = "—";
        sbEstimatedSize.textContent = "—";
        sbEstimatedVolume.textContent = "—";
        sbDetectionConfidence.textContent = "—";

        errorBox.innerHTML = `
            <div class="error-card">
                <span class="error-icon">⚠️</span>
                <div>
                    <div class="error-title">${message}</div>
                    ${detail ? `<div class="error-detail">${detail}</div>` : ""}
                </div>
            </div>
        `;
        errorBox.style.display = "block";
    }

    function hideError() {
        errorBox.style.display = "none";
        errorBox.innerHTML = "";
    }

    // =========================================================================
    // SUBMIT ANALYSIS TO BACKEND (/api/analyze)
    // =========================================================================
    async function runAnalysis(options = {}) {
        const isReanalyze = Boolean(options.reanalyze && currentSeriesData);
        if (!isReanalyze && selectedFiles.length === 0) return;

        hideError();
        emptyPanel.hidden = true;
        tabPanelsContainer.hidden = true;
        loadingPanel.hidden = false;

        const sliceCount = isReanalyze ? currentSeriesData.total_slices : selectedFiles.length;
        loadingTitle.textContent = sliceCount === 1
            ? "Analyzing 1 CT slice..."
            : `Analyzing ${sliceCount} CT slices...`;
        loadingSub.textContent = "Running U-Net segmentation & morphometric analysis";

        runBtn.disabled = true;
        runBtn.textContent = "Analyzing...";

        const formData = new FormData();
        formData.append("threshold", thresholdSlider.value);
        formData.append("patient_id", patientIdInput.value.trim());
        formData.append("patient_age", patientAgeInput.value.trim());
        formData.append("patient_sex", patientSexInput.value);

        if (isReanalyze) {
            formData.append("reanalyze", "1");
        } else {
            selectedFiles.forEach(function (file) {
                formData.append("ct_slices", file);
            });
        }

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            if (!res.ok || data.error) {
                showError(data.error || "Analysis failed. Please try again.", data.detail);
            } else {
                renderResults(data);
            }
        } catch (err) {
            showError("Analysis failed. Please try again.");
        } finally {
            loadingPanel.hidden = true;
            runBtn.disabled = selectedFiles.length === 0;
            runBtn.textContent = "Run Analysis";
        }
    }

    if (runBtn) {
        runBtn.addEventListener("click", function () {
            runAnalysis({ reanalyze: false });
        });
    }

    // =========================================================================
    // RENDER DASHBOARD RESULTS ACROSS ALL 4 TABS & SIDEBAR
    // =========================================================================
    function renderResults(data) {
        currentSeriesData = data;
        hideError();
        emptyPanel.hidden = true;
        loadingPanel.hidden = true;
        tabPanelsContainer.hidden = false;
        if (generateReportBtn) {
            generateReportBtn.disabled = false;
        }

        const threshVal = Number(data.threshold !== undefined ? data.threshold : thresholdSlider.value).toFixed(2);
        lastAnalyzed.textContent = threshVal;

        const isPositive = data.detection_status === "positive" || Number(data.peak_area_mm2) > 0;
        const confPct = (Number(data.detection_confidence || 0) * 100).toFixed(1) + "%";

        activeSliceIndex = Number(data.peak_slice_index || 0);
        if (activeSliceIndex < 0 || activeSliceIndex >= (data.slices || []).length) {
            activeSliceIndex = 0;
        }

        const peakSliceObj = (data.slices && data.slices[activeSliceIndex]) || {};

        // 1. Update Sidebar Findings
        sbTumorLocation.textContent = data.tumor_location || (isPositive ? "Left Adrenal Gland" : "No Lesion Detected");
        sbEstimatedSize.textContent = `${Number(data.estimated_size_mm || peakSliceObj.max_dimension_mm || 0).toFixed(2)} mm`;
        sbEstimatedVolume.textContent = `${Number(data.total_volume_cm3 || 0).toFixed(3)} cm³`;
        sbDetectionConfidence.textContent = confPct;

        // 2. Update Tab 1: Overview
        detConfidence.textContent = confPct;
        detThreshold.textContent = threshVal;

        if (isPositive) {
            detectionCard.classList.remove("detection-negative");
            detectionCard.classList.add("detection-positive");
            detHeadline.innerHTML = "⚠️ TUMOR DETECTED &bull; POSITIVE FINDING";
            detSubtext.textContent = "Pathological adrenal lesion detected.";
        } else {
            detectionCard.classList.remove("detection-positive");
            detectionCard.classList.add("detection-negative");
            detHeadline.innerHTML = "✅ NO TUMOR DETECTED &bull; NEGATIVE FINDING";
            detSubtext.textContent = "No pathological adrenal lesion identified above threshold.";
        }

        statSlicesAnalyzed.textContent = String(data.total_slices || 0);
        statPeakArea.textContent = `${Number(data.peak_area_mm2 || 0).toFixed(2)} mm²`;
        statTotalVolume.textContent = `${Number(data.total_volume_cm3 || 0).toFixed(3)} cm³`;
        statConfidence.textContent = confPct;

        // 3. Update active slice view (Tab 2 Segmentation + Tab 3 Measurements)
        updateActiveSliceView(activeSliceIndex);

        // 4. Update Tab 4: Profile (Always render graph)
        renderProfileChart(data);
    }

    function updateActiveSliceView(sliceIdx) {
        if (!currentSeriesData || !Array.isArray(currentSeriesData.slices) || currentSeriesData.slices.length === 0) {
            return;
        }

        const slices = currentSeriesData.slices;
        activeSliceIndex = Math.max(0, Math.min(slices.length - 1, sliceIdx));
        const s = slices[activeSliceIndex];
        const cacheBust = `?t=${Date.now()}`;

        statMaxDim.textContent = `${Number(s.max_dimension_mm || 0).toFixed(2)} mm`;
        statTumorArea.textContent = `${Number(s.area_mm2 || 0).toFixed(2)} mm²`;

        sliceCounter.textContent = `${activeSliceIndex + 1} / ${slices.length}`;
        prevSliceBtn.disabled = activeSliceIndex <= 0;
        nextSliceBtn.disabled = activeSliceIndex >= slices.length - 1;
        jumpPeakBtn.disabled = activeSliceIndex === currentSeriesData.peak_slice_index;

        segOrigImg.src = (s.original_url || "") + cacheBust;
        segMaskImg.src = (s.mask_url || "") + cacheBust;
        segOverlayImg.src = (s.overlay_url || "") + cacheBust;

        const displaySliceNum = activeSliceIndex + 1;
        const isPeak = activeSliceIndex === currentSeriesData.peak_slice_index;
        measSliceBadge.textContent = isPeak
            ? `Slice #${displaySliceNum} (Peak Slice)`
            : `Slice #${displaySliceNum}`;

        measArea.textContent = `${Number(s.area_mm2 || 0).toFixed(2)} mm² (${s.tumor_pixels || 0} px)`;
        measVolume.textContent = `${Number(currentSeriesData.total_volume_cm3 || 0).toFixed(3)} cm³`;
        measWidth.textContent = `${Number(s.width_mm || 0).toFixed(2)} mm (${s.width_px || 0} px)`;
        measHeight.textContent = `${Number(s.height_mm || 0).toFixed(2)} mm (${s.height_px || 0} px)`;
        measMaxDim.textContent = `${Number(s.max_dimension_mm || 0).toFixed(2)} mm (${s.max_dimension_px || 0} px)`;
        measCoverage.textContent = `${Number(s.coverage_percent || 0).toFixed(4)} %`;
    }

    if (prevSliceBtn) {
        prevSliceBtn.addEventListener("click", function () {
            updateActiveSliceView(activeSliceIndex - 1);
        });
    }

    if (nextSliceBtn) {
        nextSliceBtn.addEventListener("click", function () {
            updateActiveSliceView(activeSliceIndex + 1);
        });
    }

    if (jumpPeakBtn) {
        jumpPeakBtn.addEventListener("click", function () {
            if (currentSeriesData && currentSeriesData.peak_slice_index !== undefined) {
                updateActiveSliceView(currentSeriesData.peak_slice_index);
            }
        });
    }

    // =========================================================================
    // TAB 4: PROFILE — ALWAYS RENDER LINE GRAPH
    // =========================================================================
    function renderProfileChart(data) {
        if (profileChartInstance) {
            profileChartInstance.destroy();
            profileChartInstance = null;
        }

        const slices = data.slices || [];
        const labels = slices.map(function (s) { return s.slice_index + 1; }); // 1-based
        const areas = slices.map(function (s) { return s.area_mm2; });

        const canvasEl = document.getElementById("profile-chart");
        if (!canvasEl || typeof Chart === "undefined") return;

        profileChartInstance = new Chart(canvasEl, {
            type: "line",
            data: {
                labels: labels,
                datasets: [{
                    label: "Tumor Area (mm²)",
                    data: areas,
                    borderColor: "#C62828",
                    backgroundColor: "rgba(198, 40, 40, 0.08)",
                    borderWidth: 3,
                    tension: 0.3,
                    pointRadius: 6,
                    pointBackgroundColor: "#C62828",
                    pointBorderColor: "#FFFFFF",
                    pointBorderWidth: 2,
                    fill: data.total_slices > 1 // Only fill area if multiple points
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: false },
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: "CT Slice Index" },
                        grid: { color: "rgba(0,0,0,0.05)" },
                        min: 0,
                        max: data.total_slices === 1 ? 2 : undefined // Center single point
                    },
                    y: {
                        title: { display: true, text: "Tumor Area (mm²)" },
                        grid: { color: "rgba(0,0,0,0.05)" },
                        beginAtZero: true,
                        suggestedMax: data.peak_area_mm2 * 1.2 // Add headroom above point
                    }
                }
            }
        });
    }

    // =========================================================================
    // PARAMETERS & SETTINGS (Threshold Slider & Run New Analysis)
    // =========================================================================
    if (thresholdSlider) {
        thresholdSlider.addEventListener("input", function () {
            thresholdValue.textContent = Number(this.value).toFixed(2);
        });
    }

    if (runNewAnalysisBtn) {
        runNewAnalysisBtn.addEventListener("click", function () {
            const currentSliderVal = Number(thresholdSlider.value).toFixed(2);
            if (currentSeriesData && currentSliderVal !== lastAnalyzed.textContent) {
                runAnalysis({ reanalyze: true });
                return;
            }
            if (selectedFiles.length > 0 && !currentSeriesData) {
                runAnalysis({ reanalyze: false });
                return;
            }
            currentSeriesData = null;
            selectedFiles = [];
            singleInput.value = "";
            seriesInput.value = "";
            handleFiles([]);
            hideError();
            tabPanelsContainer.hidden = true;
            loadingPanel.hidden = true;
            emptyPanel.hidden = false;
            if (generateReportBtn) {
                generateReportBtn.disabled = true;
            }
            sbTumorLocation.textContent = "—";
            sbEstimatedSize.textContent = "—";
            sbEstimatedVolume.textContent = "—";
            sbDetectionConfidence.textContent = "—";
            switchTab("overview");
        });
    }
})();
