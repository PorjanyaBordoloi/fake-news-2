import { useState, useCallback } from 'react';
import { ShieldAlert, ShieldCheck, ShieldQuestion, Info, AlertTriangle } from 'lucide-react';
import { StreamResult } from '../hooks/useAgentStream';
import { translateToEnglish } from '../services/api';

interface Props {
    result: StreamResult;
}

export default function ScoreDisplay({ result }: Props) {
    const verdicts = result?.verdicts || [];
    const explanations = result?.explanations || {};
    const byClaimExplanations = explanations?.by_claim || {};
    const localizedOutput = result?.localizedOutput || null;
    const isTranslated = result?.isTranslated || false;
    const sourceLanguageName = localizedOutput?.language_name || '';
    const mediaRiskLevel = result?.mediaRiskLevel || '';
    const mediaVerdicts = result?.mediaVerdicts || [];
    const imageUrls = result?.imageUrls || [];
    const hasTamper = mediaVerdicts.some((v: any) => v.exif_tamper_flag);
    const hasOcrText = mediaVerdicts.some((v: any) => v.ocr_text);

    // Per-claim translation state: claimText → translated string (or null while loading)
    const [translatedClaims, setTranslatedClaims] = useState<Record<string, string | null>>({});
    const [translatingClaims, setTranslatingClaims] = useState<Set<string>>(new Set());
    const [showEnglish, setShowEnglish] = useState<Set<string>>(new Set());

    const handleTranslate = useCallback(async (claimText: string) => {
        if (translatedClaims[claimText] !== undefined) {
            // Already translated — just toggle display
            setShowEnglish(prev => {
                const next = new Set(prev);
                next.has(claimText) ? next.delete(claimText) : next.add(claimText);
                return next;
            });
            return;
        }
        setTranslatingClaims(prev => new Set(prev).add(claimText));
        try {
            const res = await translateToEnglish(claimText);
            setTranslatedClaims(prev => ({ ...prev, [claimText]: res.translated_text }));
            setShowEnglish(prev => new Set(prev).add(claimText));
        } catch {
            setTranslatedClaims(prev => ({ ...prev, [claimText]: null }));
        } finally {
            setTranslatingClaims(prev => { const s = new Set(prev); s.delete(claimText); return s; });
        }
    }, [translatedClaims]);

    // Detect non-ASCII (non-English) text — shows translate button on those claims
    const hasNonAscii = (text: string) => /[^\x00-\x7F]/.test(text);

    if (verdicts.length === 0 && !mediaRiskLevel && mediaVerdicts.length === 0) return null;

    // Image-only analysis: no verdicts but we have media data — show only the media panel
    if (verdicts.length === 0) {
        return (
            <div className="score-display-container">
                {mediaRiskLevel && (
                    <div className={`media-risk-banner media-risk-${mediaRiskLevel.toLowerCase()}`}>
                        <span className="media-risk-icon">{mediaRiskLevel === 'HIGH' ? '⚠' : '✓'}</span>
                        <div className="media-risk-content">
                            <div className="media-risk-title">
                                {mediaRiskLevel === 'HIGH' ? 'Media Integrity Alert' : 'Image Scan Complete'}
                            </div>
                            <div className="media-risk-detail">
                                {hasTamper && 'Image editing software detected in EXIF metadata. '}
                                {hasOcrText ? 'Text found in image — no claims detected after full pipeline. ' : 'No text found in image. '}
                                {imageUrls.length > 0 && `${imageUrls.length} image(s) scanned.`}
                            </div>
                        </div>
                    </div>
                )}
                {!mediaRiskLevel && (
                    <div className="overall-verdict-card">
                        <div className="verdict-header">
                            <ShieldQuestion color="#94a3b8" size={24} />
                            <span className="verdict-title" style={{ color: '#94a3b8' }}>No Claims Found</span>
                        </div>
                        <p className="bottom-line-text">No verifiable claims were extracted from this image.</p>
                    </div>
                )}
            </div>
        );
    }

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
            {/* Language badge — shown when input was translated by Agent 0 */}
            {isTranslated && localizedOutput && (
                <div className="language-badge">
                    Analyzed in {sourceLanguageName} · Translated via Sarvam AI
                </div>
            )}

            {/* Media Risk Banner — shown when Agent 6 detected HIGH risk */}
            {mediaRiskLevel === 'HIGH' && (
                <div className={`media-risk-banner media-risk-${mediaRiskLevel.toLowerCase()}`}>
                    <span className="media-risk-icon">⚠</span>
                    <div className="media-risk-content">
                        <div className="media-risk-title">Media Integrity Alert</div>
                        <div className="media-risk-detail">
                            {hasTamper && 'Image editing software detected in EXIF metadata. '}
                            {hasOcrText && 'Text extracted from images was included in analysis. '}
                            {imageUrls.length > 0 && `${imageUrls.length} image(s) scanned.`}
                        </div>
                    </div>
                </div>
            )}

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
                        // Prefer localized explanation; fall back to English by_claim
                        const byClaimKeys = Object.keys(byClaimExplanations);
                        const localizedClaim = localizedOutput?.by_claim?.[v.claim_text];
                        const englishClaim = byClaimExplanations[v.claim_text]
                            || (byClaimKeys.length > idx ? byClaimExplanations[byClaimKeys[idx]] : {})
                            || {};
                        const explanation = localizedClaim ?? englishClaim;
                        const isNonEnglish = hasNonAscii(v.claim_text);
                        const isShowingEnglish = showEnglish.has(v.claim_text);
                        const isTranslatingThis = translatingClaims.has(v.claim_text);
                        const englishClaimText = translatedClaims[v.claim_text];

                        return (
                            <div key={idx} className="claim-item">
                                <div className="claim-text">
                                    "{isShowingEnglish && englishClaimText ? englishClaimText : v.claim_text}"
                                    {isNonEnglish && (
                                        <button
                                            className="translate-claim-btn"
                                            onClick={() => handleTranslate(v.claim_text)}
                                            disabled={isTranslatingThis}
                                            title={isShowingEnglish ? 'Show original' : 'Translate claim to English'}
                                        >
                                            {isTranslatingThis
                                                ? 'Translating…'
                                                : isShowingEnglish
                                                ? '← Original'
                                                : '🌐 Translate'}
                                        </button>
                                    )}
                                </div>
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

                                {/* Translation note */}
                                {isTranslated && (
                                    <p className="translation-note">
                                        Translation accuracy may vary for named entities
                                    </p>
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
