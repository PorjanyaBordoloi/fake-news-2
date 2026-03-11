import { useEffect, useState } from 'react';
import { ShieldCheck, ShieldAlert, ShieldQuestion, Trash2, RefreshCw } from 'lucide-react';
import { getHistory, deleteHistoryEntry } from '../services/api';

interface HistoryEntry {
    id: string;
    url: string;
    timestamp: string;
    overall_credibility: string;
    avg_score: number;
    bottom_line: string;
    source: string;
}

interface Props {
    onAnalyze: (url: string) => void;
}

export default function HistoryPanel({ onAnalyze }: Props) {
    const [entries, setEntries] = useState<HistoryEntry[]>([]);
    const [loading, setLoading] = useState(true);

    const load = () => {
        setLoading(true);
        getHistory()
            .then(setEntries)
            .catch(console.error)
            .finally(() => setLoading(false));
    };

    useEffect(() => { load(); }, []);

    const handleDelete = async (id: string) => {
        await deleteHistoryEntry(id);
        setEntries(prev => prev.filter(e => e.id !== id));
    };

    const getColor = (v: string) => {
        const u = v.toUpperCase();
        if (u === 'CREDIBLE' || u === 'MOSTLY CREDIBLE') return '#4ade80';
        if (u === 'UNRELIABLE') return '#fca5a5';
        return '#fcd34d';
    };

    const getIcon = (v: string) => {
        const u = v.toUpperCase();
        if (u === 'CREDIBLE' || u === 'MOSTLY CREDIBLE') return <ShieldCheck color="#4ade80" size={16} />;
        if (u === 'UNRELIABLE') return <ShieldAlert color="#fca5a5" size={16} />;
        return <ShieldQuestion color="#fcd34d" size={16} />;
    };

    return (
        <div className="history-panel">
            <div className="history-header">
                <h2 className="history-title">Analysis History</h2>
                <button className="history-refresh-btn" onClick={load} title="Refresh">
                    <RefreshCw size={16} />
                </button>
            </div>

            {loading && <p className="history-empty">Loading...</p>}

            {!loading && entries.length === 0 && (
                <p className="history-empty">No analyses yet. Analyze an article to get started.</p>
            )}

            {!loading && entries.length > 0 && (
                <div className="history-list">
                    {entries.map(entry => {
                        const date = new Date(entry.timestamp).toLocaleString();
                        const displayUrl = entry.url.length > 72 ? entry.url.slice(0, 72) + '…' : entry.url;
                        const color = getColor(entry.overall_credibility);
                        return (
                            <div key={entry.id} className="history-card">
                                <div className="history-card-top">
                                    <div className="history-verdict" style={{ color }}>
                                        {getIcon(entry.overall_credibility)}
                                        <span>{entry.overall_credibility}</span>
                                        <span className="history-score">{entry.avg_score}/100</span>
                                    </div>
                                    <div className="history-actions">
                                        <span className="history-source-badge">{entry.source}</span>
                                        <span className="history-date">{date}</span>
                                        <button
                                            className="history-delete-btn"
                                            onClick={() => handleDelete(entry.id)}
                                            title="Remove"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                                <a
                                    href={entry.url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="history-url"
                                    title={entry.url}
                                >
                                    {displayUrl}
                                </a>
                                {entry.bottom_line && (
                                    <p className="history-bottom-line">{entry.bottom_line}</p>
                                )}
                                <button
                                    className="history-reanalyze-btn"
                                    onClick={() => onAnalyze(entry.url)}
                                >
                                    Re-analyze
                                </button>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
