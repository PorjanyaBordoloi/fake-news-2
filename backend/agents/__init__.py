"""
agents — Multi-agent fake news verification pipeline.

Agent 1 (claim_extraction): Extracts checkable claims from URL or text input.
Agent 2 (evidence_retrieval): Gathers supporting/refuting web evidence for claims.
Agent 3 (fact_checker): Produces final verdicts per claim using evidence + Gemini LLM.
"""
