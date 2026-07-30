/* ==========================================================================
   SmartCare Hospital AI - Dashboard Controller
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Dark/Light Theme System
    initTheme();

    // 2. Set default appointment date to today
    const dateInput = document.getElementById('appointment_date');
    if (dateInput && !dateInput.value) {
        const today = new Date().toISOString().split('T')[0];
        dateInput.value = today;
    }

    // 3. Attach Form Submission Event Listener
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', handlePrediction);
    }

    // 4. Quick Fill Demo Data Button Listener
    const demoBtn = document.getElementById('btn-fill-demo');
    if (demoBtn) {
        demoBtn.addEventListener('click', fillDemoData);
    }
});

/**
 * Dark / Light Theme Toggle System
 */
function initTheme() {
    const themeBtn = document.getElementById('theme-toggle-btn');
    const themeText = document.getElementById('theme-toggle-text');

    // Check stored theme or default to dark
    const savedTheme = localStorage.getItem('smartcare-theme') || 'dark';
    applyTheme(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const currentTheme = document.body.classList.contains('light-theme') ? 'light' : 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            applyTheme(newTheme);
            localStorage.setItem('smartcare-theme', newTheme);
        });
    }

    function applyTheme(theme) {
        if (theme === 'light') {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            document.documentElement.setAttribute('data-theme', 'light');
            if (themeText) themeText.textContent = 'Light Mode';
        } else {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
            document.documentElement.setAttribute('data-theme', 'dark');
            if (themeText) themeText.textContent = 'Dark Mode';
        }
    }
}

/**
 * Handle Prediction Form Submission
 */
async function handlePrediction(e) {
    e.preventDefault();

    const submitBtn = document.getElementById('btn-submit');
    const spinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');
    const errorAlert = document.getElementById('error-alert');

    // UI Loading State
    submitBtn.classList.add('loading');
    spinner.style.display = 'inline-block';
    btnText.textContent = 'Analyzing...';
    errorAlert.style.display = 'none';

    // Safe retrieval of optional record_id
    const recordIdEl = document.getElementById('record_id');
    const record_id = recordIdEl ? parseInt(recordIdEl.value, 10) : 1;

    // Get selected model from dropdown
    const modelSelect = document.getElementById('model-select');
    const selectedModel = modelSelect ? modelSelect.value : 'Logistic Regression';

    // Build Payload matching backend expectations
    const payload = {
        model: selectedModel,
        record_id: record_id,
        age: parseInt(document.getElementById('age').value, 10) || 40,
        gender: document.getElementById('gender').value,
        blood_group: document.getElementById('blood_group').value,
        department: document.getElementById('department').value,
        diagnosis: document.getElementById('diagnosis').value,
        appointment_date: document.getElementById('appointment_date').value,
        waiting_days: parseInt(document.getElementById('waiting_days').value, 10) || 0,
        previous_appointments: parseInt(document.getElementById('previous_appointments').value, 10) || 0,
        missed_previous_appointments: parseInt(document.getElementById('missed_previous_appointments').value, 10) || 0,
        appointment_status: document.getElementById('appointment_status').value,
        admitted: parseInt(document.getElementById('admitted').value, 10) || 0,
        room_type: document.getElementById('room_type').value,
        length_of_stay_days: parseInt(document.getElementById('length_of_stay_days').value, 10) || 0,
        previous_admissions: parseInt(document.getElementById('previous_admissions').value, 10) || 0,
        systolic_bp: parseInt(document.getElementById('systolic_bp').value, 10) || 120,
        diastolic_bp: parseInt(document.getElementById('diastolic_bp').value, 10) || 80,
        blood_sugar_mg_dl: parseInt(document.getElementById('blood_sugar_mg_dl').value, 10) || 100,
        cholesterol_mg_dl: parseInt(document.getElementById('cholesterol_mg_dl').value, 10) || 180,
        bmi: parseFloat(document.getElementById('bmi').value) || 25.0,
        lab_tests_count: parseInt(document.getElementById('lab_tests_count').value, 10) || 0,
        treatments_count: parseInt(document.getElementById('treatments_count').value, 10) || 0,
        consultation_fee_lkr: parseInt(document.getElementById('consultation_fee_lkr').value, 10) || 0,
        room_charge_lkr: parseInt(document.getElementById('room_charge_lkr').value, 10) || 0,
        lab_charge_lkr: parseInt(document.getElementById('lab_charge_lkr').value, 10) || 0,
        medicine_charge_lkr: parseInt(document.getElementById('medicine_charge_lkr').value, 10) || 0,
        total_bill_lkr: parseInt(document.getElementById('total_bill_lkr').value, 10) || 0,
        payment_status: document.getElementById('payment_status').value,
        payment_method: document.getElementById('payment_method').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (!response.ok || result.error) {
            showError(result.error || 'Server error occurred during prediction.');
        } else {
            renderResults(result);
        }
    } catch (err) {
        showError('❌ Server connection error. Please ensure Flask server is running.');
    } finally {
        // Reset UI Loading State
        submitBtn.classList.remove('loading');
        spinner.style.display = 'none';
        btnText.textContent = '🚀 Predict No-Show Risk';
    }
}

/**
 * Render Results on Dashboard
 */
function renderResults(data) {
    const resultsPanel = document.getElementById('results-panel');
    const modelBadge = document.getElementById('output-model-badge');
    const circleProgress = document.getElementById('circle-progress');
    const circlePercent = document.getElementById('circle-percent');
    const circleRiskText = document.getElementById('circle-risk-text');
    const resultStatusTitle = document.getElementById('result-status-title');
    const resultDescription = document.getElementById('result-description');
    const shapList = document.getElementById('shap-list');

    // Display Model Name
    modelBadge.textContent = `Model: ${data.model}`;

    const prob = data.probability;
    const percentVal = (prob * 100).toFixed(1);
    circlePercent.textContent = `${percentVal}%`;

    // Circular Gauge Perimeter for r=50 is 2 * PI * 50 = 314.159
    const circumference = 314.159;
    const strokeDashoffset = circumference - (prob * circumference);
    circleProgress.style.strokeDashoffset = strokeDashoffset;

    // Reset Title Classes
    resultStatusTitle.className = 'result-status-title';

    if (data.risk === 'High Risk') {
        circleProgress.style.stroke = 'var(--risk-high)';
        circleRiskText.style.color = 'var(--risk-high)';
        circleRiskText.textContent = 'HIGH';
        resultStatusTitle.classList.add('high');
        resultStatusTitle.textContent = 'High Risk 🚨';
        resultDescription.textContent = 'High Risk — Patient will not come, he needs to attend.';
    } else if (data.risk === 'Medium Risk') {
        circleProgress.style.stroke = 'var(--risk-medium)';
        circleRiskText.style.color = 'var(--risk-medium)';
        circleRiskText.textContent = 'MEDIUM';
        resultStatusTitle.classList.add('medium');
        resultStatusTitle.textContent = 'Medium Risk ⚠️';
        resultDescription.textContent = 'Medium Risk — Patient rarely comes.';
    } else {
        circleProgress.style.stroke = 'var(--risk-low)';
        circleRiskText.style.color = 'var(--risk-low)';
        circleRiskText.textContent = 'LOW';
        resultStatusTitle.classList.add('low');
        resultStatusTitle.textContent = 'Low Risk ✅';
        resultDescription.textContent = 'Low Risk — Patient generally comes for clinic.';
    }

    // Render SHAP Explainable AI Results
    shapList.innerHTML = '';
    if (data.explanation && Object.keys(data.explanation).length > 0) {
        const entries = Object.entries(data.explanation);
        const maxVal = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.001);

        entries.forEach(([feature, val]) => {
            const isPositive = val >= 0;
            const absVal = Math.abs(val);
            const fillWidth = Math.min((absVal / maxVal) * 100, 100).toFixed(1);
            const formattedName = formatFeatureName(feature);
            const formattedVal = (val >= 0 ? '+' : '') + val.toFixed(4);

            const item = document.createElement('div');
            item.className = `shap-item ${isPositive ? 'shap-positive' : 'shap-negative'}`;
            item.innerHTML = `
                <div class="shap-item-header">
                    <span class="shap-feature-name">${formattedName}</span>
                    <span class="shap-feature-value">${formattedVal}</span>
                </div>
                <div class="shap-bar-track">
                    <div class="shap-bar-fill" style="width: ${fillWidth}%;"></div>
                </div>
            `;
            shapList.appendChild(item);
        });
    }

    // Reveal Panel and Scroll smoothly into view
    resultsPanel.style.display = 'block';
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Format raw feature names for readable UI labels
 */
function formatFeatureName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
        .replace(/Lkr/g, '(LKR)')
        .replace(/Mg Dl/g, 'mg/dL')
        .replace(/Bp/g, 'BP');
}

/**
 * Display Error Alert
 */
function showError(msg) {
    const errorAlert = document.getElementById('error-alert');
    const resultsPanel = document.getElementById('results-panel');
    errorAlert.textContent = msg;
    errorAlert.style.display = 'block';
    resultsPanel.style.display = 'none';
    errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Quick Fill Demo Preset
 */
function fillDemoData() {
    const recordIdEl = document.getElementById('record_id');
    if (recordIdEl) recordIdEl.value = 101;

    document.getElementById('age').value = 45;
    document.getElementById('gender').value = 'Female';
    document.getElementById('blood_group').value = 'O+';

    document.getElementById('department').value = 'Cardiology';
    document.getElementById('diagnosis').value = 'Hypertension';
    document.getElementById('waiting_days').value = 14;
    document.getElementById('previous_appointments').value = 5;
    document.getElementById('missed_previous_appointments').value = 2;
    document.getElementById('appointment_status').value = 'Scheduled';

    document.getElementById('admitted').value = '0';
    document.getElementById('room_type').value = 'General';
    document.getElementById('length_of_stay_days').value = 0;
    document.getElementById('previous_admissions').value = 1;
    document.getElementById('systolic_bp').value = 135;
    document.getElementById('diastolic_bp').value = 88;
    document.getElementById('blood_sugar_mg_dl').value = 115;
    document.getElementById('cholesterol_mg_dl').value = 210;
    document.getElementById('bmi').value = 27.5;
    document.getElementById('lab_tests_count').value = 3;
    document.getElementById('treatments_count').value = 2;

    document.getElementById('consultation_fee_lkr').value = 2500;
    document.getElementById('room_charge_lkr').value = 0;
    document.getElementById('lab_charge_lkr').value = 1500;
    document.getElementById('medicine_charge_lkr').value = 1800;
    document.getElementById('total_bill_lkr').value = 5800;
    document.getElementById('payment_status').value = 'Paid';
    document.getElementById('payment_method').value = 'Card';
}
