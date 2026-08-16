/* SmartCare Hospital AI - Chat Controller */

document.addEventListener('DOMContentLoaded', () => {
    initTheme();

    // Set default appointment date to today
    const dateInput = document.getElementById('appointment_date');
    if (dateInput && !dateInput.value) {
        dateInput.value = new Date().toISOString().split('T')[0];
    }

    // Attach external floating predict button to form submission
    const submitBtnExt = document.getElementById('btn-submit-external');
    const form = document.getElementById('prediction-form');

    if (submitBtnExt && form) {
        submitBtnExt.addEventListener('click', (e) => {
            // Check form validity before proceeding
            if (form.checkValidity()) {
                handlePrediction(e);
            } else {
                form.reportValidity();
            }
        });
    }

    // Quick Fill Demo Data Button Listener
    const demoBtn = document.getElementById('btn-fill-demo');
    if (demoBtn) {
        demoBtn.addEventListener('click', fillDemoData);
    }
});

function initTheme() {
    const themeBtn = document.getElementById('theme-toggle-btn');
    const themeText = document.getElementById('theme-toggle-text');
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

async function handlePrediction(e) {
    e.preventDefault();

    const submitBtn = document.getElementById('btn-submit-external');
    const spinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');
    const chatContainer = document.getElementById('chat-container');
    const insertionPoint = document.getElementById('chat-insertion-point');

    // UI Loading State for Button
    submitBtn.classList.add('loading');
    spinner.style.display = 'inline-block';
    btnText.innerHTML = 'Predicting...';

    // Disable inputs while predicting
    document.querySelectorAll('#prediction-form input, #prediction-form select').forEach(el => el.disabled = true);

    const modelSelect = document.getElementById('model-select');
    const selectedModel = modelSelect ? modelSelect.value : 'Logistic Regression';

    // 1. Append User Message
    const tplUser = document.getElementById('tpl-user-msg');
    const userMsgNode = tplUser.content.cloneNode(true);
    userMsgNode.querySelector('.msg-model-name').textContent = selectedModel;
    insertionPoint.appendChild(userMsgNode);
    scrollToBottom(chatContainer);

    // 2. Append Agent Loading Message
    const tplLoading = document.getElementById('tpl-agent-loading');
    const loadingNode = tplLoading.content.cloneNode(true);
    insertionPoint.appendChild(loadingNode);
    scrollToBottom(chatContainer);

    // Slight artificial delay for smooth chat feel
    await new Promise(r => setTimeout(r, 600));

    // Gather payload
    const consultFeeEl = document.getElementById('consultation_fee_lkr');
    const consultation_fee_lkr = consultFeeEl && consultFeeEl.value !== ''
        ? parseInt(consultFeeEl.value, 10)
        : null;

    const payload = {
        model: selectedModel,
        record_id: 1,
        age: parseInt(document.getElementById('age').value, 10) || 40,
        gender: document.getElementById('gender').value,
        department: document.getElementById('department').value,
        diagnosis: document.getElementById('diagnosis').value,
        appointment_date: document.getElementById('appointment_date').value,
        waiting_days: parseInt(document.getElementById('waiting_days').value, 10) || 0,
        previous_appointments: parseInt(document.getElementById('previous_appointments').value, 10) || 0,
        missed_previous_appointments: parseInt(document.getElementById('missed_previous_appointments').value, 10) || 0,
        bmi: parseFloat(document.getElementById('bmi').value) || 25.0,
        consultation_fee_lkr: consultation_fee_lkr
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        // Remove loading message
        const loadingMsgEl = document.getElementById('loading-message');
        if (loadingMsgEl) loadingMsgEl.remove();

        if (!response.ok || result.error) {
            showErrorBubble(result.error || 'Server error occurred during prediction.', insertionPoint);
        } else {
            appendResultBubble(result, insertionPoint);
        }
    } catch (err) {
        const loadingMsgEl = document.getElementById('loading-message');
        if (loadingMsgEl) loadingMsgEl.remove();
        showErrorBubble('❌ Server connection error. Please ensure Flask server is running.', insertionPoint);
    } finally {
        // Reset UI Loading State
        submitBtn.classList.remove('loading');
        spinner.style.display = 'none';
        btnText.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Predict No-Show Risk';

        // Re-enable inputs
        document.querySelectorAll('#prediction-form input, #prediction-form select').forEach(el => el.disabled = false);

        scrollToBottom(chatContainer);
    }
}

function appendResultBubble(data, container) {
    const tplResult = document.getElementById('tpl-agent-result');
    const resultNode = tplResult.content.cloneNode(true);

    // Populate data
    resultNode.querySelector('.model-badge-inline').textContent = `Model: ${data.model}`;

    const prob = data.probability;
    const percentVal = (prob * 100).toFixed(1);
    resultNode.querySelector('.circle-percent').textContent = `${percentVal}%`;

    const circleProgress = resultNode.querySelector('.chat-circle-progress');
    const circumference = 314.159;
    circleProgress.style.strokeDashoffset = circumference - (prob * circumference);

    const circleRiskText = resultNode.querySelector('.circle-risk-text');
    const statusTitle = resultNode.querySelector('.result-status-title');
    const desc = resultNode.querySelector('.result-description');

    if (data.risk === 'High Probability of No-show') {
    circleProgress.style.stroke = 'var(--risk-high)';
    circleRiskText.style.color = 'var(--risk-high)';
    circleRiskText.textContent = 'HIGH';
    statusTitle.className = 'result-status-title high';
    statusTitle.textContent = 'High Probability of No-show';
    desc.textContent = 'Patient will likely not attend the clinic. Follow up recommended.';

} else if (data.risk === 'Moderate Probability of No-show') {
    circleProgress.style.stroke = 'var(--risk-medium)';
    circleRiskText.style.color = 'var(--risk-medium)';
    circleRiskText.textContent = 'MEDIUM';
    statusTitle.className = 'result-status-title medium';
    statusTitle.textContent = 'Moderate Probability of No-show';
    desc.textContent = 'Patient rarely attends the clinic.';

} else {
    circleProgress.style.stroke = 'var(--risk-low)';
    circleRiskText.style.color = 'var(--risk-low)';
    circleRiskText.textContent = 'LOW';
    statusTitle.className = 'result-status-title low';
    statusTitle.textContent = 'Low Probability of No-show';
    desc.textContent = 'Patient generally attends the clinic.';
}

    // SHAP Explainable AI Results
    const shapList = resultNode.querySelector('.chat-shap-list');
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

    container.appendChild(resultNode);
}

function showErrorBubble(msg, container) {
    const errorBubble = document.createElement('div');
    errorBubble.className = 'chat-message agent animate-pop-in';
    errorBubble.innerHTML = `
        <div class="chat-avatar" style="background: var(--risk-high); color: white;">
            <i class="fa-solid fa-triangle-exclamation"></i>
        </div>
        <div class="chat-bubble error-bubble">
            <p>${msg}</p>
        </div>
    `;
    container.appendChild(errorBubble);
}

function scrollToBottom(container) {
    setTimeout(() => {
        container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
        });
    }, 50); // slight delay to allow rendering
}

function formatFeatureName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
        .replace(/Lkr/g, '(LKR)')
        .replace(/Mg Dl/g, 'mg/dL')
        .replace(/Bp/g, 'BP');
}

function fillDemoData() {
    document.getElementById('age').value = 45;
    document.getElementById('gender').value = 'Female';
    document.getElementById('bmi').value = 27.5;
    document.getElementById('department').value = 'Cardiology';
    document.getElementById('diagnosis').value = 'Hypertension';
    document.getElementById('appointment_date').value = new Date().toISOString().split('T')[0];
    document.getElementById('waiting_days').value = 14;
    document.getElementById('previous_appointments').value = 5;
    document.getElementById('missed_previous_appointments').value = 2;
    document.getElementById('consultation_fee_lkr').value = 2500;
}
