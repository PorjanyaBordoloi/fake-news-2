/**
 * API Client
 *
 * Provides functions for communicating with the Fake News Detector backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Simple (non-streaming) analysis — used by Chrome extension or fallback.
 */
export async function analyzeSimple(url: string) {
    const response = await fetch(`${API_BASE_URL}/api/analyze-simple`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    return response.json();
}

export async function getHistory() {
    const response = await fetch(`${API_BASE_URL}/api/history`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function saveHistory(payload: {
    url: string;
    verdicts: any[];
    explanations: any;
    source?: string;
}) {
    const response = await fetch(`${API_BASE_URL}/api/history`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'webapp', ...payload }),
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function deleteHistoryEntry(id: string) {
    const response = await fetch(`${API_BASE_URL}/api/history/${id}`, {
        method: 'DELETE',
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

/**
 * Health check.
 */
export async function healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
}
