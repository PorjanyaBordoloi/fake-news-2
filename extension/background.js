/**
 * background.js — Extension background service worker.
 *
 * Handles communication between the content script and the popup,
 * and manages API requests to the backend.
 */

const API_URL = 'http://localhost:8000';

// Listen for messages from content script or popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'ANALYZE_URL') {
        analyzeUrl(message.url)
            .then((result) => sendResponse({ success: true, data: result }))
            .catch((error) => sendResponse({ success: false, error: error.message }));

        // Return true to indicate async response
        return true;
    }
});

async function analyzeUrl(url) {
    const response = await fetch(`${API_URL}/api/analyze-simple?user_input=${encodeURIComponent(url)}`, {
        method: 'POST',
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    return response.json();
}
