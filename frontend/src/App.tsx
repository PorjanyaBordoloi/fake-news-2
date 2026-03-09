import { useState, useEffect } from 'react';
import URLInput from './components/URLInput';
import AgentChain from './components/AgentChain';
import ScoreDisplay from './components/ScoreDisplay';
import { useAgentStream } from './hooks/useAgentStream';

function App() {
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
    const [submittedUrl, setSubmittedUrl] = useState<string>('');

    const { thoughts, isComplete, finalResult, startStream, resetStream } = useAgentStream();

    const handleSubmit = async (url: string) => {
        setSubmittedUrl(url);
        setIsAnalyzing(true);
        resetStream();
        startStream(url);
    };

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const autoUrl = params.get('url');
        if (autoUrl) {
            // Remove the URL param from address bar so it doesn't infinite loop on refresh
            window.history.replaceState({}, document.title, window.location.pathname);
            handleSubmit(autoUrl);
        }
    }, []);

    return (
        <div className="app">
            <header className="app-header">
                <div className="logo-container">
                    <span className="logo-k">K</span>
                    <span className="logo-redo">REDO</span>
                </div>
                <div className="nav-links">
                    <a href="#">Home</a>
                    <a href="#">About Project</a>
                </div>
            </header>

            <main className="app-main">
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
            </main>
        </div>
    );
}

export default App;
