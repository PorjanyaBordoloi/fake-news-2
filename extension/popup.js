const API_URL = 'http://localhost:8000';

document.getElementById('analyze-btn').addEventListener('click', async () => {
    try {
        // Get current tab URL
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const url = tab?.url || '';

        if (!url || url.startsWith('chrome://')) {
            alert('Cannot analyze special Chrome pages.');
            return;
        }

        // Show loading
        document.getElementById('not-article').classList.add('hidden');
        document.getElementById('loading').classList.remove('hidden');

        // Fetch to our backend utilizing the LangGraph pipeline
        const response = await fetch(`${API_URL}/api/analyze-simple?user_input=${encodeURIComponent(url)}`, {
            method: 'POST',
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Calculate overall score from the verdicts array returned by the Groq Agent
        const verdicts = data.verdicts || [];
        const scores = verdicts.map(v => v.truth_score || 50);
        const avgScore = scores.length > 0 ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : 0;

        let overall = 'UNVERIFIED';
        if (avgScore >= 70) overall = 'REAL';
        else if (avgScore <= 30) overall = 'FAKE';
        else if (avgScore < 70) overall = 'MISLEADING';

        // Hide loading, show result
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('result').classList.remove('hidden');

        // Populate verdict
        const container = document.getElementById('verdict-container');
        const scoreBadge = document.getElementById('score-badge');
        const verdictText = document.getElementById('verdict-text');
        const verdictDesc = document.getElementById('verdict-description');

        scoreBadge.textContent = `Score: ${avgScore}/100`;

        if (overall === 'FAKE') {
            container.className = 'verdict verdict-fake';
            verdictText.textContent = '⚠️ Likely Fake';
            verdictDesc.textContent = 'Multiple red flags detected in claims';
        } else if (overall === 'REAL') {
            container.className = 'verdict verdict-real';
            verdictText.textContent = '✅ Likely Real';
            verdictDesc.textContent = 'Claims are supported by web evidence';
        } else if (overall === 'MISLEADING') {
            container.className = 'verdict verdict-misleading';
            verdictText.textContent = '⚠️ Misleading';
            verdictDesc.textContent = 'Contains mixed or exaggerated claims';
        } else {
            container.className = 'verdict';
            verdictText.textContent = '❔ Unverified';
            verdictDesc.textContent = 'Could not be fully verified';
        }
    } catch (error) {
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('not-article').classList.remove('hidden');
        console.error('Analysis failed:', error);
        alert('Analysis failed. Make sure the backend server (http://localhost:8000) is running.');
    }
});

// View Full Analysis button → opens website
document.getElementById('view-details')?.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab?.url || '';
    if (url) {
        // Automatically start verification on the frontend dashboard
        chrome.tabs.create({ url: `http://localhost:5173?url=${encodeURIComponent(url)}` });
    }
});
