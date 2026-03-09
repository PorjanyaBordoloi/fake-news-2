import { useState, useEffect } from 'react';
import { AgentThought } from '../hooks/useAgentStream';
import { ChevronRight, ChevronDown, FileText, Sparkles } from 'lucide-react';

interface Props {
    thoughts: AgentThought[];
    isComplete: boolean;
}

export default function AgentChain({ thoughts, isComplete }: Props) {
    const [isExpanded, setIsExpanded] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);

    // Simulate terminal logs based on actual backend agent transitions
    useEffect(() => {
        setLogs([
            "--- PIPELINE INITIALIZED ---",
            "--- SCRAPING ARTICLE CONTENT VIA JINA ---",
            "--- TRUNCATING MARKDOWN TO 10000 CHARS ---",
            "--- EXTRACTING CLAIMS WITH GROQ ---"
        ]);
    }, []);

    useEffect(() => {
        const extractThought = thoughts.find(t => t.agent === 'claim_extraction');
        if (extractThought && extractThought.data) {
            const claims = extractThought.data.claims || [];
            const numClaims = claims.length;

            setLogs(prev => {
                const newLogs = [...prev];
                if (!newLogs.includes("--- CLAIM EXTRACTION COMPLETE ---")) {
                    newLogs.push("--- CLAIM EXTRACTION COMPLETE ---");
                    newLogs.push("--- EVIDENCE RETRIEVAL START ---");
                    newLogs.push(`--- Processing ${numClaims} claims (of ${numClaims} total) ---`);

                    // Add mock search strings based on real claims
                    claims.forEach((c: any) => {
                        newLogs.push(`  🔎 Tavily search: "${c.claim_text.substring(0, 70)}..."`);
                    });
                }
                return newLogs;
            });
        }
    }, [thoughts]);

    useEffect(() => {
        const evidenceThought = thoughts.find(t => t.agent === 'evidence_retrieval');
        if (evidenceThought && evidenceThought.data) {
            const evMap = evidenceThought.data.evidence_map || {};
            const numClaims = Object.keys(evMap).length;
            const totalSnippets = Object.values(evMap).reduce((acc: number, arr: any) => acc + arr.length, 0);

            setLogs(prev => {
                const newLogs = [...prev];
                if (!newLogs.includes("--- FACT CHECKER START ---")) {
                    for (let i = 0; i < numClaims; i++) {
                        newLogs.push(`  ✅ Found snippets for claim`);
                    }
                    newLogs.push(`--- EVIDENCE RETRIEVAL COMPLETE: ${totalSnippets} total snippets across ${numClaims} claims ---`);
                    newLogs.push("--- FACT CHECKER START ---");
                    newLogs.push("  🤖 Groq analyzing evidence vs claims...");
                }
                return newLogs;
            });
        }
    }, [thoughts]);

    useEffect(() => {
        if (isComplete) {
            setLogs(prev => {
                if (!prev.includes("--- FACT CHECKING COMPLETE ---")) {
                    return [...prev, "--- FACT CHECKING COMPLETE ---", "✅ Returning verdicts to user..."];
                }
                return prev;
            });
        }
    }, [isComplete]);

    return (
        <div className="agent-chain-wrapper">
            {/* Claude-like tool use accordion */}
            <div className="tool-accordion">
                <button
                    className="tool-accordion-header"
                    onClick={() => setIsExpanded(!isExpanded)}
                >
                    <div className="tool-icon-wrapper">
                        <FileText size={16} />
                    </div>
                    <span>Orchestrating verification agents...</span>
                    <div style={{ flexGrow: 1 }} />
                    {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                </button>

                {isExpanded && (
                    <div className="tool-accordion-content">
                        <div className="terminal-logs">
                            {logs.map((log, i) => (
                                <div key={i} className="log-line">{log}</div>
                            ))}
                            {!isComplete && (
                                <div className="log-line blinking-cursor">_</div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Claude-like spinner footer */}
            {!isComplete && (
                <div className="agent-status-footer">
                    <Sparkles className="sparkle-icon spinner-spin" size={20} />
                    <span className="agent-status-text">
                        Pipeline is analyzing in the background. Once it's complete, you'll see it here.
                    </span>
                </div>
            )}
        </div>
    );
}
