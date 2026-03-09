import { ShieldAlert, ShieldCheck, ShieldQuestion, Info, AlertTriangle } from 'lucide-react';
import { StreamResult } from '../hooks/useAgentStream';

interface Props {
    result: StreamResult;
}

export default function ScoreDisplay({ result }: Props) {
    const verdicts = result?.verdicts || [];
    const explanations = result?.explanations || {};
    const byClaimExplanations = explanations?.by_claim || {};

    if (verdicts.length === 0) return null;

    const scores = verdicts.map((v: any) => v.truth_score || 50);
    const averageScore = scores.length > 0 ? Math.round(scores.reduce((a: number, b: number) => a + b, 0) / scores.length) : 0;

    // Use explanation generator's overall_credibility if available, otherwise derive
    const overallCredibility = explanations?.overall_credibility || (() => {
        if (averageScore >= 70) return 'CREDIBLE';
        if (averageScore >= 50) return 'MIXED';
        if (averageScore >= 30) return 'LOW CREDIBILITY';
        return 'UNRELIABLE';
    })();

    const bottomLine = explanations?.bottom_line || '';

    const getColor = (v: string) => {
        const upper = v.toUpperCase();
        if (upper === 'SUPPORTED' || upper === 'CREDIBLE' || upper === 'MOSTLY CREDIBLE') return '#4ade80';
        if (upper === 'CONTRADICTED' || upper === 'UNRELIABLE') return '#fca5a5';
        if (upper === 'MISLEADING' || upper === 'MIXED' || upper === 'LOW CREDIBILITY') return '#fcd34d';
        return '#94a3b8';
    };

    const getIcon = (v: string) => {
        const upper = v.toUpperCase();
        if (upper === 'SUPPORTED' || upper === 'CREDIBLE' || upper === 'MOSTLY CREDIBLE') return <ShieldCheck color="#4ade80" size={24} />;
        if (upper === 'CONTRADICTED' || upper === 'UNRELIABLE') return <ShieldAlert color="#fca5a5" size={24} />;
        return <ShieldQuestion color={getColor(v)} size={24} />;
    };

    return (
        <div className="score-display-container">
            {/* Overall Credibility Card */}
            <div className="overall-verdict-card" style={{ borderLeft: `4px solid ${getColor(overallCredibility)}` }}>
                <div className="verdict-header">
                    {getIcon(overallCredibility)}
                    <span className="verdict-title" style={{ color: getColor(overallCredibility) }}>
                        {overallCredibility}
                    </span>
                    <span className="verdict-score-badge">
                        {averageScore} / 100 Trust Score
                    </span>
                </div>
                {bottomLine && (
                    <p className="bottom-line-text">{bottomLine}</p>
                )}
            </div>

            {/* Detailed Claim Analysis */}
            <div className="detailed-verdicts">
                <h4 className="details-heading">Detailed Claim Analysis</h4>
                <div className="claims-list">
                    {verdicts.map((v: any, idx: number) => {
                        // Try exact key match first, then fall back to index-based match
                        // (LLMs sometimes slightly rephrase claim_text in their output)
                        const byClaimKeys = Object.keys(byClaimExplanations);
                        const explanation = byClaimExplanations[v.claim_text]
                            || (byClaimKeys.length > idx ? byClaimExplanations[byClaimKeys[idx]] : {})
                            || {};
                        return (
                            <div key={idx} className="claim-item">
                                <div className="claim-text">"{v.claim_text}"</div>
                                <div className="claim-assessment" style={{ color: getColor(v.verdict) }}>
                                    {v.verdict} ({v.truth_score}/100)
                                    {v.confidence_level && (
                                        <span className="confidence-badge"> · {v.confidence_level}</span>
                                    )}
                                </div>

                                {/* Plain English explanation from Agent 5 */}
                                {explanation.plain_english && (
                                    <div className="plain-english-block">
                                        <p>{explanation.plain_english}</p>
                                    </div>
                                )}

                                {/* Confidence Statement */}
                                {explanation.confidence_statement && (
                                    <div className="meta-note">
                                        <Info size={14} /> {explanation.confidence_statement}
                                    </div>
                                )}

                                {/* Source Quality Note */}
                                {explanation.source_quality_note && (
                                    <div className="meta-note">
                                        <ShieldCheck size={14} /> {explanation.source_quality_note}
                                    </div>
                                )}

                                {/* Reader Advisory */}
                                {explanation.reader_advisory && (
                                    <div className="reader-advisory">
                                        <AlertTriangle size={14} /> {explanation.reader_advisory}
                                    </div>
                                )}

                                {/* Evidence Gaps */}
                                {explanation.evidence_gaps_plain && (
                                    <div className="meta-note evidence-gap">
                                        <ShieldQuestion size={14} /> {explanation.evidence_gaps_plain}
                                    </div>
                                )}

                                {/* Fallback to raw explanation if no Agent 5 data */}
                                {!explanation.plain_english && v.explanation && (
                                    <div className="claim-explanation">{v.explanation}</div>
                                )}

                                {/* Citations */}
                                {v.citations && v.citations.length > 0 && (
                                    <ul className="citation-list">
                                        {v.citations.map((cit: string, cIdx: number) => (
                                            <li key={cIdx}><a href={cit} target="_blank" rel="noreferrer">[{cIdx + 1}] Source</a></li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
