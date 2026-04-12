const defaults = {
radius:14.12, texture:19.29, perimeter:91.97, area:654.8,
smoothness:0.096, compactness:0.104, concavity:0.088,
concave_points:0.048, symmetry:0.181, fractal_dimension:0.062
};

const API_URL = 'http://localhost:8000/prediction';

const form        = document.getElementById('predictionForm');
const submitBtn   = document.getElementById('submitBtn');
const resetBtn    = document.getElementById('resetBtn');
const idleState   = document.getElementById('idle-state');
const resultPanel = document.getElementById('result-panel');
const errorPanel  = document.getElementById('errorPanel');
const errorMsg    = document.getElementById('errorMsg');

resetBtn.addEventListener('click', () => {
for (const [key, val] of Object.entries(defaults)) {
    const el = document.getElementById(key);
    if (el) el.value = val;
}
});

form.addEventListener('submit', async (e) => {
e.preventDefault();
const payload = {};
for (const key of Object.keys(defaults)) {
    const val = parseFloat(document.getElementById(key).value);
    if (isNaN(val)) { alert(`Please enter a valid number for "${key}".`); return; }
    payload[key] = val;
}

submitBtn.disabled = true;
submitBtn.innerHTML = '<span class="spinner-ring"></span>Analyzing&hellip;';
resultPanel.style.display = 'none';
errorPanel.style.display  = 'none';
idleState.style.display   = 'none';

try {
    const res = await fetch(API_URL, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `HTTP ${res.status}`); }
    renderResult(await res.json());
} catch (err) {
    errorPanel.style.display = 'block';
    errorMsg.textContent = err.message;
} finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = 'Run Classification';
}
});

function renderResult(data) {
const now = new Date();
document.getElementById('resultTimestamp').textContent =
    now.toLocaleDateString('en-US',{year:'numeric',month:'short',day:'numeric'})
    + ' · ' + now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});

document.getElementById('resultLabel').textContent     = data.result;
document.getElementById('confidencePill').textContent  = `Confidence: ${data.confiability.toFixed(1)}%`;
document.getElementById('pctMalignant').textContent    = `${data.probability_malignant.toFixed(1)}%`;
document.getElementById('pctBenign').textContent       = `${data.probability_benign.toFixed(1)}%`;

const bM = document.getElementById('barMalignant');
const bB = document.getElementById('barBenign');
bM.style.width = '0%'; bB.style.width = '0%';
requestAnimationFrame(() => requestAnimationFrame(() => {
    bM.style.width = `${data.probability_malignant}%`;
    bB.style.width = `${data.probability_benign}%`;
}));
resultPanel.style.display = 'block';
}