import { useState, useEffect, useRef } from 'react';
import { AgentThought } from '../hooks/useAgentStream';

interface Props {
    thoughts: AgentThought[];
    isComplete: boolean;
}

interface LogEntry {
    text: string;
    type: 'status' | 'detail' | 'success' | 'search' | 'score' | 'verdict' | 'final';
}

export default function AgentChain({ thoughts, isComplete }: Props) {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [visibleCount, setVisibleCount] = useState(0);
    const bottomRef = useRef<HTMLDivElement>(null);
    const prevLogCount = useRef(0);

    // Initialize pipeline logs
    useEffect(() => {
        setLogs([
            { text: "Initializing pipeline...", type: "status" },
            { text: "Scraping article content via Jina Reader", type: "status" },
            { text: "Truncating content for analysis", type: "detail" },
            { text: "Extracting claims with Groq LLM", type: "status" },
        ]);
        setVisibleCount(0);
    }, []);

    // Agent 1: Claim Extraction
    useEffect(() => {
        const t = thoughts.find(t => t.agent === 'claim_extraction');
        if (t?.data) {
            const claims = t.data.claims || [];
            setLogs(prev => {
                if (prev.some(l => l.text.startsWith("Extracted"))) return prev;
                return [
                    ...prev,
                    { text: `Extracted ${claims.length} checkable claims`, type: "success" },
                    { text: "Starting evidence retrieval across the web", type: "status" },
                    ...claims.map((c: any) => ({
                        text: `Searching: "${(c.claim_text || c.claim || '').substring(0, 80)}..."`,
                        type: "search" as const,
                    })),
                ];
            });
        }
    }, [thoughts]);

    // Agent 2: Evidence Retrieval
    useEffect(() => {
        const t = thoughts.find(t => t.agent === 'evidence_retrieval');
        if (t?.data) {
            const evMap = t.data.evidence_map || {};
            const total = Object.values(evMap).reduce((acc: number, arr: any) => acc + arr.length, 0);
            setLogs(prev => {
                if (prev.some(l => l.text.startsWith("Evidence collected"))) return prev;
                const newLogs: LogEntry[] = [];
                for (const [, snippets] of Object.entries(evMap)) {
                    newLogs.push({ text: `Found ${(snippets as any[]).length} evidence snippets`, type: "success" });
                }
                newLogs.push({ text: `Evidence collected: ${total} snippets across ${Object.keys(evMap).length} claims`, type: "success" });
                newLogs.push({ text: "Scoring source credibility by domain tier", type: "status" });
                return [...prev, ...newLogs];
            });
        }
    }, [thoughts]);

    // Agent 3: Source Credibility
    useEffect(() => {
        const t = thoughts.find(t => t.agent === 'source_credibility');
        if (t?.data) {
            const credMap = t.data.credibility_map || {};
            setLogs(prev => {
                if (prev.some(l => l.text.startsWith("Sources scored"))) return prev;
                const newLogs: LogEntry[] = [];
                for (const [, entries] of Object.entries(credMap)) {
                    const typed = entries as any[];
                    if (typed.length > 0) {
                        const top = typed[0];
                        newLogs.push({
                            text: `Top source: ${top.domain} — ${top.score}/100 (${top.label})`,
                            type: "score",
                        });
                    }
                }
                newLogs.push({ text: `Sources scored and re-ranked for ${Object.keys(credMap).length} claims`, type: "success" });
                newLogs.push({ text: "Running fact-check reasoning with Groq", type: "status" });
                return [...prev, ...newLogs];
            });
        }
    }, [thoughts]);

    // Agent 4: Fact Checker
    useEffect(() => {
        const t = thoughts.find(t => t.agent === 'fact_checker');
        if (t?.data) {
            const verdicts = t.data.verdicts || [];
            setLogs(prev => {
                if (prev.some(l => l.text.startsWith("All claims fact-checked"))) return prev;
                const newLogs: LogEntry[] = [];
                for (const v of verdicts) {
                    const conf = v.confidence_level || 'MEDIUM';
                    const emoji = v.verdict === 'SUPPORTED' ? '✓' : v.verdict === 'CONTRADICTED' ? '✗' : '~';
                    newLogs.push({
                        text: `${emoji} ${v.verdict} (${v.truth_score}/100, ${conf}) — "${(v.claim_text || '').substring(0, 60)}..."`,
                        type: "verdict",
                    });
                }
                newLogs.push({ text: `All claims fact-checked: ${verdicts.length} verdicts`, type: "success" });
                newLogs.push({ text: "Generating plain-English explanations", type: "status" });
                return [...prev, ...newLogs];
            });
        }
    }, [thoughts]);

    // Agent 5: Explanation Generator
    useEffect(() => {
        const t = thoughts.find(t => t.agent === 'explanation_generator');
        if (t?.data) {
            const expl = t.data.explanations || {};
            setLogs(prev => {
                if (prev.some(l => l.text.startsWith("Credibility:"))) return prev;
                const newLogs: LogEntry[] = [];
                if (expl.overall_credibility) {
                    newLogs.push({ text: `Credibility: ${expl.overall_credibility}`, type: "final" });
                }
                if (expl.bottom_line) {
                    newLogs.push({ text: expl.bottom_line, type: "final" });
                }
                return [...prev, ...newLogs];
            });
        }
    }, [thoughts]);

    // Pipeline complete
    useEffect(() => {
        if (isComplete) {
            setLogs(prev => {
                if (prev.some(l => l.text === "Pipeline complete — results ready.")) return prev;
                return [...prev, { text: "Pipeline complete — results ready.", type: "success" }];
            });
        }
    }, [isComplete]);

    // Animate new logs appearing one by one
    useEffect(() => {
        if (logs.length > prevLogCount.current) {
            const newEntries = logs.length - prevLogCount.current;
            let i = 0;
            const interval = setInterval(() => {
                setVisibleCount(prev => prev + 1);
                i++;
                if (i >= newEntries) clearInterval(interval);
            }, 120);
            prevLogCount.current = logs.length;
            return () => clearInterval(interval);
        }
    }, [logs]);

    // Auto-scroll as new lines appear
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, [visibleCount]);

    const visibleLogs = logs.slice(0, visibleCount);

    return (
        <div className="agent-stream">
            {visibleLogs.map((log, i) => (
                <div
                    key={i}
                    className={`stream-line stream-${log.type} stream-enter`}
                    style={{ animationDelay: '0ms' }}
                >
                    <span className="stream-indicator">{getIndicator(log.type)}</span>
                    <span className="stream-text">{log.text}</span>
                </div>
            ))}
            {!isComplete && visibleCount >= logs.length && (
                <div className="stream-line stream-thinking">
                    <span className="stream-indicator">
                        <span className="thinking-dots">
                            <span>.</span><span>.</span><span>.</span>
                        </span>
                    </span>
                    <span className="stream-text stream-text-thinking">Thinking</span>
                </div>
            )}
            <div ref={bottomRef} />
        </div>
    );
}

function getIndicator(type: string): string {
    switch (type) {
        case 'status': return '→';
        case 'detail': return '·';
        case 'success': return '✓';
        case 'search': return '⌕';
        case 'score': return '◆';
        case 'verdict': return '⬤';
        case 'final': return '★';
        default: return '·';
    }
}
