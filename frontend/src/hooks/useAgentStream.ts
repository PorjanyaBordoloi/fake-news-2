import { useState, useCallback } from 'react';

export interface AgentThought {
    agent: string;
    status: string;
    data?: any;
}

export function useAgentStream() {
    const [thoughts, setThoughts] = useState<AgentThought[]>([]);
    const [isComplete, setIsComplete] = useState(false);
    const [finalResult, setFinalResult] = useState<any>(null);

    const startStream = useCallback((userInput: string) => {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const eventSource = new EventSource(
            `${apiUrl}/api/analyze-stream?user_input=${encodeURIComponent(userInput)}`
        );

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.status === 'complete') {
                    setIsComplete(true);
                    eventSource.close();
                } else if (data.agent) {
                    setThoughts((prev) => [...prev, data]);
                    // If it's the final step, store the verdicts
                    if (data.agent === 'fact_checker' && data.status === 'success') {
                        setFinalResult(data.data);
                    }
                }
            } catch (err) {
                console.error("Error parsing event data:", err);
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
        };

        return () => eventSource.close();
    }, []);

    const resetStream = useCallback(() => {
        setThoughts([]);
        setIsComplete(false);
        setFinalResult(null);
    }, []);

    return { thoughts, isComplete, finalResult, startStream, resetStream };
}
