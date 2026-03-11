const API_URL = 'http://localhost:8000';

// ─── Analyze logic ──────────────────────────────────────
async function runAnalysis(inputString) {
    if (!inputString || inputString.startsWith('chrome://') || inputString.startsWith('arc://') || inputString.startsWith('about:')) {
        showError('Cannot analyze browser internal pages or empty inputs. Please enter a valid URL or text.');
        return;
    }

    // Save for deep analysis button
    const viewDetailsBtn = document.getElementById('view-details');
    if (viewDetailsBtn) viewDetailsBtn.dataset.url = inputString;

    // Show loading
    document.getElementById('not-article').classList.add('hidden');
    document.getElementById('error-state').classList.add('hidden');
    document.getElementById('result').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');

    const container = document.getElementById('pipeline-steps');
    container.innerHTML = '';
    const etaEl = document.getElementById('pipeline-eta');
    if (etaEl) etaEl.textContent = 'Running fast analysis (top claim only)...';

    let currentStepEl = null;

    const appendStep = (text) => {
        if (currentStepEl) {
            currentStepEl.classList.remove('step-active');
            currentStepEl.classList.add('step-done');
            currentStepEl.querySelector('.step-indicator').textContent = '✓';
        }
        currentStepEl = document.createElement('div');
        currentStepEl.className = 'pipeline-step step-active';
        currentStepEl.innerHTML = `
            <span class="step-indicator">●</span>
            <span>${text}</span>
        `;
        container.appendChild(currentStepEl);
    };

    appendStep("Initializing connection...");

    const es = new EventSource(`${API_URL}/api/analyze-stream?user_input=${encodeURIComponent(inputString)}&top_n=1`);

    let collectedVerdicts = [];
    let collectedExplanations = {};
    let isTranslated = false;
    let localizedBottomLine = '';

    es.onmessage = async (e) => {
        const data = JSON.parse(e.data);

        if (data.type === 'agent_log') {
            appendStep(data.message);
            return;
        }

        if (data.status === 'complete') {
            es.close();
            if (currentStepEl) {
                currentStepEl.classList.remove('step-active');
                currentStepEl.classList.add('step-done');
                currentStepEl.querySelector('.step-indicator').textContent = '✓';
            }
            const doneDiv = document.createElement('div');
            doneDiv.className = 'pipeline-step step-done';
            doneDiv.innerHTML = `<span class="step-indicator">✓</span><span>Analysis complete</span>`;
            container.appendChild(doneDiv);
            if (etaEl) etaEl.textContent = '';

            // Render final result
            await new Promise(r => setTimeout(r, 400));

            const verdicts = collectedVerdicts;
            const explanations = collectedExplanations;
            const scores = verdicts.map(v => v.truth_score || 50);
            const avgScore = scores.length > 0 ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : 0;

            let overall = explanations.overall_credibility || '';
            let bottomLine = (isTranslated && localizedBottomLine) ? localizedBottomLine : (explanations.bottom_line || '');

            if (!overall) {
                if (avgScore >= 70) overall = 'CREDIBLE';
                else if (avgScore <= 30) overall = 'UNRELIABLE';
                else overall = 'MIXED';
            }

            document.getElementById('loading').classList.add('hidden');
            document.getElementById('result').classList.remove('hidden');

            const verdictContainer = document.getElementById('verdict-container');
            const scoreBadge = document.getElementById('score-badge');
            const verdictText = document.getElementById('verdict-text');
            const verdictDesc = document.getElementById('verdict-description');

            scoreBadge.textContent = `${avgScore}/100`;

            const upperOverall = overall.toUpperCase();
            if (upperOverall === 'UNRELIABLE' || upperOverall === 'LOW CREDIBILITY') {
                verdictContainer.className = 'verdict verdict-fake';
                verdictText.textContent = '⚠️ Likely Fake';
                verdictDesc.textContent = bottomLine || 'Multiple red flags detected in claims.';
            } else if (upperOverall === 'CREDIBLE' || upperOverall === 'MOSTLY CREDIBLE') {
                verdictContainer.className = 'verdict verdict-real';
                verdictText.textContent = '✅ Likely Real';
                verdictDesc.textContent = bottomLine || 'Claims are supported by credible web evidence.';
            } else if (upperOverall === 'MIXED' || upperOverall === 'MISLEADING') {
                verdictContainer.className = 'verdict verdict-misleading';
                verdictText.textContent = '⚠️ Mixed / Misleading';
                verdictDesc.textContent = bottomLine || 'Contains mixed or exaggerated claims.';
            } else {
                verdictContainer.className = 'verdict';
                verdictText.textContent = '❔ Unverified';
                verdictDesc.textContent = bottomLine || 'Could not be fully verified with available evidence.';
            }
            return;
        }

        if (!data.agent) return;

        if (data.agent === 'agent0_pre') {
            appendStep('Detecting language...');
        } else if (data.agent === 'claim_extraction') {
            appendStep('Extracting checkable claims...');
        } else if (data.agent === 'evidence_retrieval') {
            appendStep('Searching the web for evidence...');
        } else if (data.agent === 'source_credibility') {
            appendStep('Scoring source credibility...');
        } else if (data.agent === 'fact_checker') {
            appendStep('Reasoning through evidence...');
            // Collect verdicts from the fact_checker node output
            if (data.data && data.data.verdicts) {
                collectedVerdicts = data.data.verdicts;
            }
        } else if (data.agent === 'explanation_generator') {
            appendStep('Generating explanations...');
            if (data.data && data.data.explanations) {
                collectedExplanations = data.data.explanations;
            }
        } else if (data.agent === 'agent0_post') {
            appendStep('Finalizing explanations...');
            isTranslated = true;
            if (data.data && data.data.localized_output && data.data.localized_output.bottom_line) {
                localizedBottomLine = data.data.localized_output.bottom_line;
            }
        }
    };

    es.onerror = () => {
        es.close();
        showError('Analysis stream interrupted or failed.');
    };
}

document.getElementById('analyze-btn').addEventListener('click', async () => {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const url = tab?.url || '';
        runAnalysis(url);
    } catch (e) {
        showError('Failed to get current page URL.');
    }
});

document.getElementById('analyze-query-btn')?.addEventListener('click', () => {
    const query = document.getElementById('query-input')?.value.trim();
    if (query) {
        runAnalysis(query);
    } else {
        showError('Please enter a query or URL.');
    }
});

// ─── Error handling ─────────────────────────────────────
function showError(message) {
    document.getElementById('not-article').classList.add('hidden');
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('result').classList.add('hidden');
    document.getElementById('error-state').classList.remove('hidden');
    document.getElementById('error-msg').textContent = message;
}

document.getElementById('retry-btn')?.addEventListener('click', () => {
    document.getElementById('error-state').classList.add('hidden');
    document.getElementById('not-article').classList.remove('hidden');
});

// ─── View Deep Analysis → opens website ──────────────────
document.getElementById('view-details')?.addEventListener('click', () => {
    const url = document.getElementById('view-details').dataset.url || '';
    if (url) {
        chrome.tabs.create({ url: `http://localhost:5173?url=${encodeURIComponent(url)}` });
    }
});
