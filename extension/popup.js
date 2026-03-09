const API_URL = 'http://localhost:8000';

// Pipeline stages shown to the user during analysis
const PIPELINE_STAGES = [
    { text: "Scraping article content...", delay: 0 },
    { text: "Extracting checkable claims with Groq", delay: 3000 },
    { text: "Searching the web for evidence", delay: 7000 },
    { text: "Scoring source credibility by domain tier", delay: 13000 },
    { text: "Reasoning through evidence (2-call fact-check)", delay: 18000 },
    { text: "Generating plain-English explanations", delay: 28000 },
];

// ─── Show current page URL ────────────────────────────────
(async () => {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const url = tab?.url || '';
        const urlEl = document.getElementById('current-url');
        if (urlEl && url) {
            // Show truncated URL
            const display = url.length > 60 ? url.substring(0, 57) + '...' : url;
            urlEl.textContent = display;
        }
    } catch (_) { /* ignore */ }
})();

// ─── Pipeline progress animation ──────────────────────────
let stageTimers = [];

function showPipelineProgress() {
    const container = document.getElementById('pipeline-steps');
    container.innerHTML = '';

    // Clear any old timers
    stageTimers.forEach(t => clearTimeout(t));
    stageTimers = [];

    PIPELINE_STAGES.forEach((stage, idx) => {
        const timer = setTimeout(() => {
            // Mark previous step as done
            const prevStep = container.querySelector('.step-active');
            if (prevStep) {
                prevStep.classList.remove('step-active');
                prevStep.classList.add('step-done');
                prevStep.querySelector('.step-indicator').textContent = '✓';
            }

            // Add new step
            const div = document.createElement('div');
            div.className = 'pipeline-step step-active';
            div.innerHTML = `
                <span class="step-indicator">●</span>
                <span>${stage.text}</span>
            `;
            container.appendChild(div);

            // Update ETA
            const remaining = Math.max(0, Math.round((35 - stage.delay / 1000)));
            const etaEl = document.getElementById('pipeline-eta');
            if (etaEl) {
                etaEl.textContent = remaining > 5
                    ? `~${remaining}s remaining`
                    : 'Almost done...';
            }
        }, stage.delay);

        stageTimers.push(timer);
    });
}

function clearPipelineProgress() {
    stageTimers.forEach(t => clearTimeout(t));
    stageTimers = [];

    // Mark all existing steps as done
    const container = document.getElementById('pipeline-steps');
    container.querySelectorAll('.step-active').forEach(el => {
        el.classList.remove('step-active');
        el.classList.add('step-done');
        el.querySelector('.step-indicator').textContent = '✓';
    });

    // Add completion step
    const div = document.createElement('div');
    div.className = 'pipeline-step step-done';
    div.innerHTML = `<span class="step-indicator">✓</span><span>Analysis complete</span>`;
    container.appendChild(div);

    const etaEl = document.getElementById('pipeline-eta');
    if (etaEl) etaEl.textContent = '';
}

// ─── Analyze button ─────────────────────────────────────
document.getElementById('analyze-btn').addEventListener('click', async () => {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const url = tab?.url || '';

        if (!url || url.startsWith('chrome://') || url.startsWith('arc://') || url.startsWith('about:')) {
            showError('Cannot analyze browser internal pages. Navigate to a news article first.');
            return;
        }

        // Show loading
        document.getElementById('not-article').classList.add('hidden');
        document.getElementById('error-state').classList.add('hidden');
        document.getElementById('result').classList.add('hidden');
        document.getElementById('loading').classList.remove('hidden');

        // Start progress animation
        showPipelineProgress();

        // Call backend
        const response = await fetch(`${API_URL}/api/analyze-simple?user_input=${encodeURIComponent(url)}`, {
            method: 'POST',
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();

        // Stop progress, show completion
        clearPipelineProgress();

        // Small delay so user sees "Analysis complete" before result
        await new Promise(r => setTimeout(r, 600));

        // Parse response
        const explanations = data.explanations || {};
        const verdicts = data.verdicts || [];
        const scores = verdicts.map(v => v.truth_score || 50);
        const avgScore = scores.length > 0
            ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length)
            : 0;

        let overall = explanations.overall_credibility || 'UNVERIFIED';
        if (!explanations.overall_credibility) {
            if (avgScore >= 70) overall = 'CREDIBLE';
            else if (avgScore <= 30) overall = 'UNRELIABLE';
            else overall = 'MIXED';
        }

        const bottomLine = explanations.bottom_line || '';

        // Show result
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('result').classList.remove('hidden');

        const container = document.getElementById('verdict-container');
        const scoreBadge = document.getElementById('score-badge');
        const verdictText = document.getElementById('verdict-text');
        const verdictDesc = document.getElementById('verdict-description');

        scoreBadge.textContent = `${avgScore}/100`;

        const upperOverall = overall.toUpperCase();
        if (upperOverall === 'UNRELIABLE' || upperOverall === 'LOW CREDIBILITY') {
            container.className = 'verdict verdict-fake';
            verdictText.textContent = '⚠️ Likely Fake';
            verdictDesc.textContent = bottomLine || 'Multiple red flags detected in claims.';
        } else if (upperOverall === 'CREDIBLE' || upperOverall === 'MOSTLY CREDIBLE') {
            container.className = 'verdict verdict-real';
            verdictText.textContent = '✅ Likely Real';
            verdictDesc.textContent = bottomLine || 'Claims are supported by credible web evidence.';
        } else if (upperOverall === 'MIXED' || upperOverall === 'MISLEADING') {
            container.className = 'verdict verdict-misleading';
            verdictText.textContent = '⚠️ Mixed / Misleading';
            verdictDesc.textContent = bottomLine || 'Contains mixed or exaggerated claims.';
        } else {
            container.className = 'verdict';
            verdictText.textContent = '❔ Unverified';
            verdictDesc.textContent = bottomLine || 'Could not be fully verified with available evidence.';
        }

    } catch (error) {
        console.error('Analysis failed:', error);
        stageTimers.forEach(t => clearTimeout(t));
        stageTimers = [];
        document.getElementById('loading').classList.add('hidden');
        showError('Analysis failed. Make sure the backend server is running on localhost:8000.');
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
document.getElementById('view-details')?.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab?.url || '';
    if (url) {
        chrome.tabs.create({ url: `http://localhost:5173?url=${encodeURIComponent(url)}` });
    }
});
