const API_BASE_URL = "http://localhost:8000";

const PRESETS = [
  { name: "Swords", a: 1040, b: 1334 },
  { name: "Knife → Sword", a: 1410, b: 1355 },
  { name: "Potions", a: 796, b: 700 },
];

const PER_PAGE = 24;

const state = {
  idxA: 0,
  idxB: 10,
  page: 0,
  seed: Math.floor(Math.random() * 1_000_000),
  totalPages: 1,
  selectedId: null,
  t: 0.5,
  useFilter: true,
  alphaThresh: 0.5,
  colorSteps: 12,
  lastBlendBlob: null,
};

const el = {
  presetRow: document.getElementById("preset-row"),
  galleryGrid: document.getElementById("gallery-grid"),
  pageIndicator: document.getElementById("page-indicator"),
  prevPage: document.getElementById("prev-page"),
  nextPage: document.getElementById("next-page"),
  shuffleBtn: document.getElementById("shuffle-btn"),
  assignHint: document.getElementById("assign-hint"),
  assignA: document.getElementById("assign-a"),
  assignB: document.getElementById("assign-b"),
  previewA: document.getElementById("preview-a"),
  previewB: document.getElementById("preview-b"),
  idA: document.getElementById("id-a"),
  idB: document.getElementById("id-b"),
  tSlider: document.getElementById("t-slider"),
  tValue: document.getElementById("t-value"),
  useFilter: document.getElementById("use-filter"),
  alphaSlider: document.getElementById("alpha-slider"),
  alphaValue: document.getElementById("alpha-value"),
  colorSlider: document.getElementById("color-slider"),
  colorValue: document.getElementById("color-value"),
  resultImg: document.getElementById("result-img"),
  resultT: document.getElementById("result-t"),
  loadingOverlay: document.getElementById("loading-overlay"),
  downloadBtn: document.getElementById("download-btn"),
  apiStatus: document.getElementById("api-status"),
};

function debounce(fn, delay) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function spriteImageUrl(id) {
  return `${API_BASE_URL}/sprites/${id}`;
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    const data = await res.json();
    if (data.status === "ok" && data.model_loaded) {
      el.apiStatus.textContent = `backend ready — ${data.total_sprites} sprites loaded`;
      el.apiStatus.className = "ok";
    } else {
      el.apiStatus.textContent = "backend reachable but not fully loaded";
      el.apiStatus.className = "error";
    }
  } catch (err) {
    el.apiStatus.textContent = `cannot reach backend at ${API_BASE_URL}`;
    el.apiStatus.className = "error";
  }
}

function renderPresets() {
  el.presetRow.innerHTML = "";
  PRESETS.forEach((preset) => {
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = preset.name;
    btn.addEventListener("click", () => {
      state.idxA = preset.a;
      state.idxB = preset.b;
      state.t = 0.5;
      el.tSlider.value = "0.5";
      el.tValue.textContent = "0.50";
      updatePreviews();
      requestBlend();
    });
    el.presetRow.appendChild(btn);
  });
}

async function fetchAndRenderGallery() {
  el.galleryGrid.innerHTML = `<div style="color: var(--text-dim); grid-column: 1 / -1;">loading…</div>`;

  try {
    const url = `${API_BASE_URL}/sprites?page=${state.page}&per_page=${PER_PAGE}&seed=${state.seed}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    state.totalPages = data.total_pages;
    el.pageIndicator.textContent = `Page ${data.page + 1} of ${data.total_pages}`;

    el.galleryGrid.innerHTML = "";
    data.sprites.forEach(({ id, thumbnail }) => {
      const thumb = document.createElement("div");
      thumb.className = "thumb";
      thumb.dataset.id = id;
      if (id === state.selectedId) thumb.classList.add("selected");

      const img = document.createElement("img");
      img.src = thumbnail;
      img.alt = `Sprite ${id}`;
      thumb.appendChild(img);

      thumb.addEventListener("click", () => {
        state.selectedId = id;
        el.assignHint.textContent = `Selected: sprite ${id}`;
        document
          .querySelectorAll(".thumb.selected")
          .forEach((t) => t.classList.remove("selected"));
        thumb.classList.add("selected");
      });

      el.galleryGrid.appendChild(thumb);
    });
  } catch (err) {
    el.galleryGrid.innerHTML = `<div style="color: var(--danger); grid-column: 1 / -1;">Failed to load gallery: ${err.message}</div>`;
  }
}

function setupGalleryControls() {
  el.prevPage.addEventListener("click", () => {
    if (state.page > 0) {
      state.page -= 1;
      fetchAndRenderGallery();
    }
  });

  el.nextPage.addEventListener("click", () => {
    if (state.page < state.totalPages - 1) {
      state.page += 1;
      fetchAndRenderGallery();
    }
  });

  el.shuffleBtn.addEventListener("click", () => {
    state.seed = Math.floor(Math.random() * 1_000_000);
    state.page = 0;
    fetchAndRenderGallery();
  });

  el.assignA.addEventListener("click", () => {
    if (state.selectedId === null) return;
    state.idxA = state.selectedId;
    updatePreviews();
    requestBlend();
  });

  el.assignB.addEventListener("click", () => {
    if (state.selectedId === null) return;
    state.idxB = state.selectedId;
    updatePreviews();
    requestBlend();
  });
}

function updatePreviews() {
  el.previewA.src = spriteImageUrl(state.idxA);
  el.previewB.src = spriteImageUrl(state.idxB);
  el.idA.textContent = state.idxA;
  el.idB.textContent = state.idxB;
}

function setupBlendControls() {
  el.tSlider.addEventListener("input", () => {
    state.t = parseFloat(el.tSlider.value);
    el.tValue.textContent = state.t.toFixed(2);
    debouncedBlend();
  });

  el.useFilter.addEventListener("change", () => {
    state.useFilter = el.useFilter.checked;
    requestBlend();
  });

  el.alphaSlider.addEventListener("input", () => {
    state.alphaThresh = parseFloat(el.alphaSlider.value);
    el.alphaValue.textContent = state.alphaThresh.toFixed(2);
    debouncedBlend();
  });

  el.colorSlider.addEventListener("input", () => {
    state.colorSteps = parseInt(el.colorSlider.value, 10);
    el.colorValue.textContent = state.colorSteps;
    debouncedBlend();
  });
}

async function requestBlend() {
  el.loadingOverlay.classList.add("visible");

  const body = {
    id_a: state.idxA,
    id_b: state.idxB,
    t: state.t,
    use_filter: state.useFilter,
    alpha_thresh: state.alphaThresh,
    color_steps: state.colorSteps,
  };

  try {
    const res = await fetch(`${API_BASE_URL}/blend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    state.lastBlendBlob = blob;

    const objectUrl = URL.createObjectURL(blob);
    if (el.resultImg.dataset.objectUrl) {
      URL.revokeObjectURL(el.resultImg.dataset.objectUrl);
    }
    el.resultImg.src = objectUrl;
    el.resultImg.dataset.objectUrl = objectUrl;
    el.resultT.textContent = state.t.toFixed(2);
    el.downloadBtn.disabled = false;
  } catch (err) {
    console.error("Blend request failed:", err);
    el.apiStatus.textContent = `blend failed: ${err.message}`;
    el.apiStatus.className = "error";
  } finally {
    el.loadingOverlay.classList.remove("visible");
  }
}

const debouncedBlend = debounce(requestBlend, 150);

function setupDownload() {
  el.downloadBtn.addEventListener("click", () => {
    if (!state.lastBlendBlob) return;
    const url = URL.createObjectURL(state.lastBlendBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `blended_sprite_t${state.t.toFixed(2)}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });
}

async function init() {
  el.downloadBtn.disabled = true;
  renderPresets();
  setupGalleryControls();
  setupBlendControls();
  setupDownload();
  updatePreviews();

  await checkHealth();
  await fetchAndRenderGallery();
  await requestBlend();
}

init();
