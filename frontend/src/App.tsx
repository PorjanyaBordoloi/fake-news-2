import { useState, useCallback, useRef, useEffect } from 'react';
import URLInput from './components/URLInput';
import AgentChain from './components/AgentChain';
import ScoreDisplay from './components/ScoreDisplay';
import HistoryPanel from './components/HistoryPanel';
import { useAgentStream } from './hooks/useAgentStream';
import { saveHistory } from './services/api';

function App() {
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
    const [submittedUrl, setSubmittedUrl] = useState<string>('');
    const [showHistory, setShowHistory] = useState(false);

    const { thoughts, isComplete, finalResult, startStream, resetStream } = useAgentStream();

    const handleSubmit = useCallback((url: string) => {
        setSubmittedUrl(url);
        setIsAnalyzing(true);
        setShowHistory(false);
        resetStream();
        startStream(url);
    }, [resetStream, startStream]);

    // Auto-start analysis if ?url= param is present (from Chrome extension handoff)
    const hasAutoStarted = useRef(false);
    useEffect(() => {
        if (hasAutoStarted.current) return;
        const params = new URLSearchParams(window.location.search);
        const autoUrl = params.get('url');
        if (autoUrl) {
            hasAutoStarted.current = true;
            window.history.replaceState({}, document.title, window.location.pathname);
            handleSubmit(autoUrl);
        }
    }, [handleSubmit]);

    // Save to history when stream completes
    const hasSavedRef = useRef(false);
    useEffect(() => {
        if (isComplete && finalResult && submittedUrl && !hasSavedRef.current) {
            hasSavedRef.current = true;
            saveHistory({
                url: submittedUrl,
                verdicts: finalResult.verdicts,
                explanations: finalResult.explanations,
            }).catch(console.error);
        }
    }, [isComplete, finalResult, submittedUrl]);

    // Reset save guard on new analysis
    useEffect(() => {
        if (!isComplete) hasSavedRef.current = false;
    }, [isComplete]);

    return (
        <div className="app">
            <header className="app-header">
                <div className="logo-container">
                    <span className="logo-k">K</span>
                    <span className="logo-redo">REDO</span>
                </div>
                <div className="nav-links">
                    <a href="#" onClick={e => { e.preventDefault(); setShowHistory(false); }}>Home</a>
                    <a href="#" onClick={e => { e.preventDefault(); setShowHistory(true); }}>History</a>
                    <a href="#">About Project</a>
                </div>
            </header>

            <main className="app-main">
                {showHistory ? (
                    <HistoryPanel onAnalyze={handleSubmit} />
                ) : (
                    <>
                        {!isAnalyzing && !submittedUrl && (
                            <h1 className="hero-title">Lets Verify !</h1>
                        )}

                        <URLInput onSubmit={handleSubmit} isLoading={isAnalyzing && !isComplete} />

                        {(isAnalyzing || thoughts.length > 0) && (
                            <div className="chat-container">
                                {/* User Message Bubble */}
                                <div className="chat-message user-message">
                                    <div className="chat-bubble" style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>
                                        Please verify this content to feed into the truth engine:<br /><br />
                                        {submittedUrl.startsWith('http') ? (
                                            <a href={submittedUrl} target="_blank" rel="noreferrer" style={{ color: '#fff', textDecoration: 'underline' }}>
                                                {submittedUrl}
                                            </a>
                                        ) : (
                                            <span style={{ fontStyle: 'italic', opacity: 0.9 }}>
                                                "{submittedUrl.length > 300 ? submittedUrl.substring(0, 300) + '...' : submittedUrl}"
                                            </span>
                                        )}
                                    </div>
                                </div>

                                {/* Assistant Message Bubble */}
                                <div className="chat-message assistant-message">
                                    <div className="chat-bubble assistant-bg">
                                        <p className="assistant-greeting">
                                            Perfect! Let me orchestrate the multi-agent pipeline to extract claims, retrieve evidence, and fact-check this article.
                                        </p>

                                        <AgentChain thoughts={thoughts} isComplete={isComplete} />

                                        {isComplete && finalResult && (
                                            <div className="final-result-wrapper fade-in">
                                                <ScoreDisplay result={finalResult} />
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                )}
            </main>
        </div>
    );
}

export default App;
