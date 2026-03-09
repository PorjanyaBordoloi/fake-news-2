"""
agents — Multi-agent fake news verification pipeline (5 agents).

Agent 1 (claim_extraction):      Extracts checkable claims from URL or text input via Groq.
Agent 2 (evidence_retrieval):    Gathers supporting/refuting web evidence via Tavily Search.
Agent 3 (source_credibility):    Scores and re-ranks evidence by domain credibility tier.
Agent 4 (fact_checker):          Produces verdicts per claim using 2-call Groq reasoning.
Agent 5 (explanation_generator): Generates plain-English explanations via Groq.
"""
