/**
 * app.js - TransactionGuard frontend
 * ====================================
 * Renders the prediction feature form and wires it to POST /api/predict.
 * Only runs when the predict-form element is present (models trained).
 */

"use strict";

// Feature names in the exact order the backend expects
const FEATURES = [
  "Time",
  ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`),
  "Amount",
];

// Sample transaction values for the "Fill sample values" button
// These are made-up representative numbers – NOT from the dataset
const SAMPLE_VALUES = {
  Time: 10000,
  V1: -1.3598,  V2: -0.0728,  V3: 2.5363,  V4: 1.3782,
  V5: -0.3383,  V6: 0.4624,   V7: 0.2396,  V8: 0.0987,
  V9: 0.3638,   V10: 0.0908,  V11: -0.5516, V12: -0.6178,
  V13: -0.9914, V14: -0.3112, V15: 1.4682,  V16: -0.4704,
  V17: 0.2079,  V18: 0.0258,  V19: 0.4040,  V20: 0.2514,
  V21: -0.0183, V22: 0.2778,  V23: -0.1105, V24: 0.0669,
  V25: 0.1285,  V26: -0.1891, V27: 0.1336,  V28: -0.0210,
  Amount: 149.62,
};

// ── Build feature input grid ───────────────────────────────────────────────

function buildForm() {
  const container = document.getElementById("feature-inputs");
  if (!container) return;

  FEATURES.forEach((name) => {
    const group = document.createElement("div");
    group.className = "feature-input-group";

    const label = document.createElement("label");
    label.htmlFor = `feat-${name}`;
    label.textContent = name;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.id = `feat-${name}`;
    input.name = name;
    input.placeholder = "0";
    input.required = true;

    group.appendChild(label);
    group.appendChild(input);
    container.appendChild(group);
  });
}

// ── Helpers ────────────────────────────────────────────────────────────────

function getInputValues() {
  const row = {};
  let valid = true;
  FEATURES.forEach((name) => {
    const el = document.getElementById(`feat-${name}`);
    const val = parseFloat(el.value);
    if (isNaN(val)) {
      el.style.borderColor = "#a84d5c";
      valid = false;
    } else {
      el.style.borderColor = "";
      row[name] = val;
    }
  });
  return valid ? row : null;
}

function fillSample() {
  FEATURES.forEach((name) => {
    const el = document.getElementById(`feat-${name}`);
    if (el) {
      el.value = SAMPLE_VALUES[name] ?? 0;
      el.style.borderColor = "";
    }
  });
}

function clearForm() {
  FEATURES.forEach((name) => {
    const el = document.getElementById(`feat-${name}`);
    if (el) {
      el.value = "";
      el.style.borderColor = "";
    }
  });
  hideResult();
}

function showResult(html) {
  const box = document.getElementById("predict-result");
  box.innerHTML = html;
  box.classList.remove("hidden");
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

function hideResult() {
  const box = document.getElementById("predict-result");
  box.innerHTML = "";
  box.classList.add("hidden");
}

// ── Render prediction response ─────────────────────────────────────────────

function renderResult(data) {
  const rf = data.random_forest;
  const iso = data.isolation_forest;
  const deviations = data.most_unusual_features || [];

  const rfFlagged = rf.predicted_class === 1;
  const isoFlagged = iso.is_anomaly;

  const rfCard = `
    <div class="result-card ${rfFlagged ? "flagged" : "clear"}">
      <h4>Random Forest</h4>
      <div class="result-verdict">${rfFlagged ? "⚠ Flag for review" : "✓ Not flagged"}</div>
      <div class="result-score">Fraud score: ${rf.fraud_score.toFixed(6)}</div>
      <div class="result-note">${rf.note}</div>
    </div>`;

  const isoCard = `
    <div class="result-card ${isoFlagged ? "flagged" : "clear"}">
      <h4>Isolation Forest</h4>
      <div class="result-verdict">${isoFlagged ? "⚠ Anomaly flagged" : "✓ Not flagged as anomaly"}</div>
      <div class="result-score">Anomaly score: ${iso.anomaly_score.toFixed(6)}</div>
      <div class="result-note">${iso.note}</div>
    </div>`;

  const deviationRows = deviations
    .map(
      (d) =>
        `<div class="deviation-row">
          <span>${d.feature}</span>
          <span>value ${d.your_value} &nbsp;|&nbsp; typical ${d.typical_value} &nbsp;|&nbsp; z = ${d.absolute_z_score}</span>
        </div>`
    )
    .join("");

  const devsSection = deviations.length
    ? `<div class="deviations-list">
        <h4>Most unusual feature values (vs training set)</h4>
        ${deviationRows}
      </div>`
    : "";

  showResult(`
    <div class="result-grid">${rfCard}${isoCard}</div>
    ${devsSection}
    <p class="result-disclaimer">${data.disclaimer}</p>
  `);
}

function renderError(msg) {
  showResult(`<p class="error-msg">⚠ ${msg}</p>`);
}

// ── Form submit handler ────────────────────────────────────────────────────

async function handleSubmit(e) {
  e.preventDefault();
  const values = getInputValues();
  if (!values) {
    renderError("Please fill in all feature fields with numeric values.");
    return;
  }

  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  btn.textContent = "Screening…";
  hideResult();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });

    const data = await response.json();

    if (!response.ok) {
      renderError(data.error || `Server error ${response.status}`);
    } else {
      renderResult(data);
    }
  } catch (err) {
    renderError("Network error – is the Flask server running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Screen transaction";
  }
}

// ── Initialise ────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  if (!form) return; // Models not trained – form not rendered

  buildForm();

  form.addEventListener("submit", handleSubmit);

  document.getElementById("fill-sample-btn")
    ?.addEventListener("click", fillSample);

  document.getElementById("clear-btn")
    ?.addEventListener("click", clearForm);
});
