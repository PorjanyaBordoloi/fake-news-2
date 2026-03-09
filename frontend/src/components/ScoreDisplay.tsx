import { ShieldAlert, ShieldCheck, ShieldQuestion } from 'lucide-react';

interface Props {
    result: any;
}

export default function ScoreDisplay({ result }: Props) {
    if (!result || !result.verdicts) return null;

    const scores = result.verdicts.map((v: any) => v.truth_score || 50);
    const averageScore = scores.length > 0 ? Math.round(scores.reduce((a: number, b: number) => a + b, 0) / scores.length) : 0;

    let overall = 'UNVERIFIED';
    if (averageScore >= 70) overall = 'SUPPORTED';
    else if (averageScore <= 30) overall = 'CONTRADICTED';
    else if (averageScore < 70) overall = 'MISLEADING';

    const getColor = (v: string) => {
        if (v === 'SUPPORTED') return '#4ade80';
        if (v === 'CONTRADICTED') return '#fca5a5';
        if (v === 'MISLEADING') return '#fcd34d';
        return '#94a3b8';
    };

    const getIcon = (v: string) => {
        if (v === 'SUPPORTED') return <ShieldCheck color="#4ade80" size={24} />;
        if (v === 'CONTRADICTED') return <ShieldAlert color="#fca5a5" size={24} />;
        return <ShieldQuestion color={getColor(v)} size={24} />;
    };

    return (
        <div className="score-display-container">
            <div className="overall-verdict-card" style={{ borderLeft: `4px solid ${getColor(overall)}` }}>
                <div className="verdict-header">
                    {getIcon(overall)}
                    <span className="verdict-title" style={{ color: getColor(overall) }}>
                        {overall}
                    </span>
                    <span className="verdict-score-badge">
                        {averageScore} / 100 Trust Score
                    </span>
                </div>
            </div>

            <div className="detailed-verdicts">
                <h4 className="details-heading">Detailed Claim Analysis</h4>
                <div className="claims-list">
                    {result.verdicts.map((v: any, idx: number) => (
                        <div key={idx} className="claim-item">
                            <div className="claim-text">"{v.claim_text}"</div>
                            <div className="claim-assessment" style={{ color: getColor(v.verdict) }}>
                                {v.verdict} ({v.truth_score}/100)
                            </div>
                            <div className="claim-explanation">{v.explanation}</div>
                            {v.citations && v.citations.length > 0 && (
                                <ul className="citation-list">
                                    {v.citations.map((cit: string, cIdx: number) => (
                                        <li key={cIdx}><a href={cit} target="_blank" rel="noreferrer">[{cIdx + 1}] Source</a></li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
