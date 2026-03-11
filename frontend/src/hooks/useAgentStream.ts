import { useState, useCallback, useRef } from 'react';

export interface AgentThought {
    agent: string;
    status: string;
    data?: any;
}

export interface StreamResult {
    verdicts: any[];
    explanations: any;
    localizedOutput?: any;
    isTranslated?: boolean;
    sourceLanguage?: string;
}

export function useAgentStream() {
    const [thoughts, setThoughts] = useState<AgentThought[]>([]);
    const [isComplete, setIsComplete] = useState(false);
    const [finalResult, setFinalResult] = useState<StreamResult | null>(null);

    // Track the EventSource so we can clean it up
    const eventSourceRef = useRef<EventSource | null>(null);

    const startStream = useCallback((userInput: string) => {
        // Close any existing connection first
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }

        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const eventSource = new EventSource(
            `${apiUrl}/api/analyze-stream?user_input=${encodeURIComponent(userInput)}`
        );
        eventSourceRef.current = eventSource;

        let collectedVerdicts: any[] = [];
        let collectedExplanations: any = null;
        let collectedLocalizedOutput: any = null;
        let collectedIsTranslated: boolean = false;
        let collectedSourceLanguage: string = '';

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.status === 'complete') {
                    setIsComplete(true);
                    setFinalResult({
                        verdicts: collectedVerdicts,
                        explanations: collectedExplanations,
                        localizedOutput: collectedLocalizedOutput,
                        isTranslated: collectedIsTranslated,
                        sourceLanguage: collectedSourceLanguage,
                    });
                    eventSource.close();
                    eventSourceRef.current = null;
                } else if (data.type === 'agent_log') {
                    // Agent 0 SSE log events — push as synthetic thought for AgentChain
                    const logThought = { agent: 'agent_log', status: 'success', data: { symbol: data.symbol, message: data.message } };
                    setThoughts((prev) => [...prev, logThought]);
                } else if (data.agent) {
                    setThoughts((prev) => [...prev, data]);

                    // Collect verdicts from fact_checker
                    if (data.agent === 'fact_checker' && data.data?.verdicts) {
                        collectedVerdicts = data.data.verdicts;
                    }
                    // Collect explanations from explanation_generator
                    if (data.agent === 'explanation_generator' && data.data?.explanations) {
                        collectedExplanations = data.data.explanations;
                    }
                    // Collect multilingual data from Agent 0
                    if (data.agent === 'agent0_pre' && data.data?.is_translated) {
                        collectedIsTranslated = true;
                        collectedSourceLanguage = data.data.source_language || '';
                    }
                    if (data.agent === 'agent0_post' && data.data?.localized_output) {
                        collectedLocalizedOutput = data.data.localized_output;
                    }
                }
            } catch (err) {
                console.error("Error parsing event data:", err);
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            eventSourceRef.current = null;
        };
    }, []);

    const resetStream = useCallback(() => {
        // Clean up any existing connection when resetting
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        setThoughts([]);
        setIsComplete(false);
        setFinalResult(null);
    }, []);

    return { thoughts, isComplete, finalResult, startStream, resetStream };
}
