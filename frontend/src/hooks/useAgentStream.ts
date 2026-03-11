import { useState, useCallback, useRef } from 'react';

export interface AgentThought {
    agent: string;
    status: string;
    data?: any;
}

export interface MediaVerdict {
    url: string;
    exif_tamper_flag: boolean;
    ocr_text?: string;
}

export interface StreamResult {
    verdicts: any[];
    explanations: any;
    localizedOutput?: any;
    isTranslated?: boolean;
    sourceLanguage?: string;
    mediaRiskLevel?: string;
    imageUrls?: string[];
    mediaVerdicts?: MediaVerdict[];
}

export function useAgentStream() {
    const [thoughts, setThoughts] = useState<AgentThought[]>([]);
    const [isComplete, setIsComplete] = useState(false);
    const [finalResult, setFinalResult] = useState<StreamResult | null>(null);

    // Track the EventSource so we can clean it up
    const eventSourceRef = useRef<EventSource | null>(null);
    // Track in-flight fetch-based image streams
    const abortControllerRef = useRef<AbortController | null>(null);

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
        let collectedMediaRiskLevel: string = '';
        let collectedImageUrls: string[] = [];
        let collectedMediaVerdicts: MediaVerdict[] = [];

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
                        mediaRiskLevel: collectedMediaRiskLevel || undefined,
                        imageUrls: collectedImageUrls.length > 0 ? collectedImageUrls : undefined,
                        mediaVerdicts: collectedMediaVerdicts.length > 0 ? collectedMediaVerdicts : undefined,
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
                    if (data.agent === 'image_integrity') {
                        if (data.data?.media_risk_level) collectedMediaRiskLevel = data.data.media_risk_level;
                        if (data.data?.image_urls) collectedImageUrls = data.data.image_urls;
                        if (data.data?.media_verdicts) collectedMediaVerdicts = data.data.media_verdicts;
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

    const startImageStream = useCallback((file: File) => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        const controller = new AbortController();
        abortControllerRef.current = controller;

        setThoughts([]);
        setIsComplete(false);
        setFinalResult(null);

        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const formData = new FormData();
        formData.append('file', file);

        let collectedVerdicts: any[] = [];
        let collectedExplanations: any = null;
        let collectedLocalizedOutput: any = null;
        let collectedIsTranslated: boolean = false;
        let collectedSourceLanguage: string = '';
        let collectedMediaRiskLevel: string = '';
        let collectedImageUrls: string[] = [];
        let collectedMediaVerdicts: MediaVerdict[] = [];

        const processEvent = (data: any) => {
            if (data.status === 'complete') {
                setIsComplete(true);
                setFinalResult({
                    verdicts: collectedVerdicts,
                    explanations: collectedExplanations,
                    localizedOutput: collectedLocalizedOutput,
                    isTranslated: collectedIsTranslated,
                    sourceLanguage: collectedSourceLanguage,
                    mediaRiskLevel: collectedMediaRiskLevel || undefined,
                    imageUrls: collectedImageUrls.length > 0 ? collectedImageUrls : undefined,
                    mediaVerdicts: collectedMediaVerdicts.length > 0 ? collectedMediaVerdicts : undefined,
                });
            } else if (data.type === 'agent_log') {
                const logThought = { agent: 'agent_log', status: 'success', data: { symbol: data.symbol, message: data.message } };
                setThoughts((prev) => [...prev, logThought]);
            } else if (data.type === 'image_result') {
                const d = data.data || {};
                if (d.media_risk_level) collectedMediaRiskLevel = d.media_risk_level;
                if (d.image_urls) collectedImageUrls = d.image_urls;
                if (d.media_verdicts) collectedMediaVerdicts = d.media_verdicts;
            } else if (data.agent) {
                setThoughts((prev) => [...prev, data]);
                if (data.agent === 'fact_checker' && data.data?.verdicts) {
                    collectedVerdicts = data.data.verdicts;
                }
                if (data.agent === 'explanation_generator' && data.data?.explanations) {
                    collectedExplanations = data.data.explanations;
                }
                if (data.agent === 'agent0_pre' && data.data?.is_translated) {
                    collectedIsTranslated = true;
                    collectedSourceLanguage = data.data.source_language || '';
                }
                if (data.agent === 'agent0_post' && data.data?.localized_output) {
                    collectedLocalizedOutput = data.data.localized_output;
                }
            }
        };

        fetch(`${apiUrl}/api/analyze-image-stream`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        }).then(response => {
            if (!response.ok) {
                console.error(`[ImageStream] Server error: ${response.status}`);
                setIsComplete(true);
                return;
            }
            if (!response.body) {
                console.error('[ImageStream] No response body');
                setIsComplete(true);
                return;
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            const read = () => {
                reader.read().then(({ done, value }) => {
                    if (controller.signal.aborted) return;
                    if (done) {
                        // Flush any remaining buffered data
                        if (buffer.startsWith('data: ')) {
                            try { processEvent(JSON.parse(buffer.slice(6))); } catch { /* ignore */ }
                        }
                        return;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try { processEvent(JSON.parse(line.slice(6))); } catch { /* ignore */ }
                        }
                    }
                    read();
                }).catch(() => { /* stream ended or aborted */ });
            };
            read();
        }).catch(() => { /* fetch failed or aborted */ });
    }, []);

    const resetStream = useCallback(() => {
        // Clean up any existing connection when resetting
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        setThoughts([]);
        setIsComplete(false);
        setFinalResult(null);
    }, []);

    return { thoughts, isComplete, finalResult, startStream, startImageStream, resetStream };
}
