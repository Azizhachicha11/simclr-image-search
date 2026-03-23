// ── DOM Elements ─────────────────────────────────────────────────────────────
const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("file-input");
const browseBtn      = document.getElementById("browse-btn");
const queryPreview   = document.getElementById("query-preview");
const queryImg       = document.getElementById("query-img");
const queryName      = document.getElementById("query-name");
const queryMeta      = document.getElementById("query-meta");
const searchBtn      = document.getElementById("search-btn");
const topkSlider     = document.getElementById("topk-slider");
const topkValue      = document.getElementById("topk-value");
const loadingOverlay = document.getElementById("loading-overlay");
const resultsSection = document.getElementById("results-section");
const resultsGrid    = document.getElementById("results-grid");
const resultsCount   = document.getElementById("results-count");
const errorMsg       = document.getElementById("error-msg");

let selectedFile = null;

// ── Top-K Slider ─────────────────────────────────────────────────────────────
topkSlider.addEventListener("input", () => {
    topkValue.textContent = topkSlider.value;
});

// ── Drag & Drop ──────────────────────────────────────────────────────────────
dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

// ── File Picker ──────────────────────────────────────────────────────────────
browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
});

dropzone.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// ── Handle Selected File ─────────────────────────────────────────────────────
function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        showError("Please upload an image file (JPG, PNG, etc.)");
        return;
    }

    selectedFile = file;
    hideError();

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        queryImg.src = e.target.result;
        queryName.textContent = file.name;
        queryMeta.textContent = `${(file.size / 1024).toFixed(1)} KB · ${file.type}`;
        queryPreview.classList.add("visible");
    };
    reader.readAsDataURL(file);

    // Auto-search
    performSearch();
}

// ── Search ───────────────────────────────────────────────────────────────────
searchBtn.addEventListener("click", () => {
    if (selectedFile) performSearch();
});

async function performSearch() {
    if (!selectedFile) return;

    hideError();
    resultsSection.classList.remove("visible");
    loadingOverlay.classList.add("visible");

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("top_k", topkSlider.value);

    try {
        const response = await fetch("/search", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Search failed");
        }

        renderResults(data);
    } catch (err) {
        showError(`Search failed: ${err.message}`);
    } finally {
        loadingOverlay.classList.remove("visible");
    }
}

// ── Render Results ───────────────────────────────────────────────────────────
function renderResults(data) {
    resultsGrid.innerHTML = "";
    const results = data.results;

    resultsCount.textContent = `${results.length} matches from ${data.total_indexed.toLocaleString()} indexed images`;

    results.forEach((item, i) => {
        const card = document.createElement("div");
        card.className = "result-card";
        card.style.animationDelay = `${i * 0.08}s`;

        const simPercent = Math.max(0, Math.min(100, item.similarity * 100));

        card.innerHTML = `
            <div class="result-card__img-wrapper">
                <img class="result-card__img" src="data:image/png;base64,${item.thumbnail}" alt="${item.class}">
                <div class="result-card__rank">${item.rank}</div>
            </div>
            <div class="result-card__body">
                <div class="result-card__class">${item.class}</div>
                <div class="result-card__sim">
                    <div class="sim-bar">
                        <div class="sim-bar__fill" style="width: ${simPercent}%"></div>
                    </div>
                    <span class="sim-value">${item.similarity.toFixed(3)}</span>
                </div>
            </div>
        `;

        resultsGrid.appendChild(card);
    });

    resultsSection.classList.add("visible");
}

// ── Error Handling ───────────────────────────────────────────────────────────
function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.add("visible");
}

function hideError() {
    errorMsg.classList.remove("visible");
}
