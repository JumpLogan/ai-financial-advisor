"""
Financial Intelligence Assistant - Module 3: Analysis Tools
File: src/analysis/financial_analyzer.py

PURPOSE: Production-grade financial analysis using RAG + structured prompts
INTERVIEW IMPACT: Shows you can build complete AI-powered workflows

CONCEPTS YOU'LL LEARN:
- Structured output generation from LLMs
- Multi-step reasoning chains
- Domain-specific prompt engineering
- Production error handling and monitoring

UPDATED: Compatible with LangChain 0.2+
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class FinancialAnalyzer:
    """
    High-level financial analysis tools powered by RAG
    
    ═══════════════════════════════════════════════════════════════════
    DESIGN PATTERN: Specialized analyzers for different use cases
    ═══════════════════════════════════════════════════════════════════
    
    Why not one generic "analyze" function?
    1. Better prompts: Task-specific instructions
    2. Structured outputs: Predictable JSON formats
    3. Validation: Domain-specific checks
    4. Composability: Chain analyzers together
    
    INTERVIEW TIP: Discuss how you designed the API for product use
    """
    
    def __init__(self, rag_engine, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize analyzer with RAG engine
        
        PATTERN: Dependency injection
        - FinancialAnalyzer depends on RAGEngine
        - Passed in constructor (not created internally)
        - Benefits: Testing, flexibility, single responsibility
        """
        self.rag = rag_engine
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0)
    
    def generate_executive_summary(self, company: str) -> Dict:
        """
        Generate high-level summary of company performance
        
        USE CASE: First page of investment memo, board presentation
        
        ═══════════════════════════════════════════════════════════════
        STRUCTURED OUTPUT TECHNIQUE
        ═══════════════════════════════════════════════════════════════
        
        Problem: LLM outputs are text, we need structured data
        Solution: Prompt for JSON format, parse the response
        
        INTERVIEW DISCUSSION POINTS:
        - Why JSON? Easy to parse, integrate with other systems
        - Alternatives: XML, YAML, or LLM function calling
        - Error handling: What if LLM doesn't follow format?
        """
        
        prompt = f"""Based on the available financial documents for {company}, 
generate a comprehensive executive summary. Structure your response as JSON:

{{
  "company": "{company}",
  "period_covered": "timeframe of most recent data",
  "key_metrics": {{
    "revenue": "latest revenue with YoY change",
    "profitability": "net income or operating margin",
    "growth_rate": "revenue or earnings growth"
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "challenges": ["challenge 1", "challenge 2"],
  "outlook": "forward-looking statement",
  "analyst_rating": "Buy/Hold/Sell with brief reasoning"
}}

Provide specific numbers and cite filing dates when available.
Return ONLY the JSON object, no additional text."""
        
        try:
            # Query RAG system
            result = self.rag.query(prompt, return_sources=True)
            
            # Parse JSON from response
            # Note: In production, add retry logic if parsing fails
            answer_text = result['answer']
            
            # Extract JSON (LLM might add markdown formatting)
            json_start = answer_text.find('{')
            json_end = answer_text.rfind('}') + 1
            json_str = answer_text[json_start:json_end]
            
            summary = json.loads(json_str)
            summary['generated_at'] = datetime.now().isoformat()
            summary['sources'] = result['sources']
            
            return summary
            
        except json.JSONDecodeError as e:
            # Fallback: Return raw text if JSON parsing fails
            return {
                "error": "Failed to parse JSON",
                "raw_response": result['answer'],
                "company": company
            }
    
    def analyze_risk_factors(self, company: str) -> Dict:
        """
        Extract and categorize risk factors from filings
        
        USE CASE: Risk management, due diligence, portfolio monitoring
        
        ═══════════════════════════════════════════════════════════════
        CLASSIFICATION TASK
        ═══════════════════════════════════════════════════════════════
        
        LLMs are excellent at:
        - Extracting specific information (e.g., risks from 10-K)
        - Categorizing text (e.g., operational vs market risk)
        - Summarizing dense content
        
        INTERVIEW QUESTION: "How would you evaluate this?"
        ANSWER: 
        - Manual review of sample outputs (qualitative)
        - Compare against human-labeled test set (quantitative)
        - A/B test in production (user engagement metrics)
        """
        
        prompt = f"""Analyze risk factors for {company} from the financial documents.
Extract and categorize the risks into:
- Market risks (competition, demand changes)
- Operational risks (supply chain, technology)
- Financial risks (debt, liquidity, currency)
- Regulatory risks (compliance, legal)

Return as JSON:
{{
  "company": "{company}",
  "risk_categories": {{
    "market": ["risk 1", "risk 2"],
    "operational": ["risk 1", "risk 2"],
    "financial": ["risk 1", "risk 2"],
    "regulatory": ["risk 1", "risk 2"]
  }},
  "overall_risk_level": "Low/Medium/High",
  "key_concerns": ["most critical risk", "second most critical"],
  "mitigating_factors": ["what company is doing to address risks"]
}}

Focus on material risks that could impact financial performance.
Return ONLY the JSON object."""
        
        result = self.rag.query(prompt)
        
        try:
            json_start = result['answer'].find('{')
            json_end = result['answer'].rfind('}') + 1
            json_str = result['answer'][json_start:json_end]
            risk_analysis = json.loads(json_str)
            risk_analysis['generated_at'] = datetime.now().isoformat()
            return risk_analysis
        except:
            return {"error": "Parsing failed", "raw": result['answer']}
    
    def comparative_analysis(self, companies: List[str]) -> Dict:
        """
        Compare multiple companies across key dimensions
        
        USE CASE: Sector analysis, investment decision, market research
        
        ═══════════════════════════════════════════════════════════════
        MULTI-DOCUMENT REASONING
        ═══════════════════════════════════════════════════════════════
        
        Challenge: Need info from multiple company filings
        Solution: RAG retrieves from all indexed documents
        
        ADVANCED TECHNIQUE: 
        - Query each company separately, then synthesize
        - More accurate but higher latency and cost
        - Trade-off: Accuracy vs speed vs cost (common interview topic!)
        """
        
        companies_str = ", ".join(companies)
        
        prompt = f"""Compare the following companies: {companies_str}

Analyze across these dimensions:
1. Financial performance (revenue, profitability, growth)
2. Market position (competitive advantages, market share)
3. Innovation & R&D (product pipeline, technology)
4. Financial health (debt levels, cash flow, liquidity)
5. Management quality (strategy execution, capital allocation)

Return as JSON:
{{
  "companies_analyzed": [list of companies],
  "comparison_date": "date",
  "metrics_comparison": {{
    "revenue_growth": {{"company1": "X%", "company2": "Y%"}},
    "profit_margin": {{"company1": "X%", "company2": "Y%"}},
    "debt_to_equity": {{"company1": "X", "company2": "Y"}}
  }},
  "competitive_positioning": {{
    "company1": "brief positioning statement",
    "company2": "brief positioning statement"
  }},
  "investment_recommendation": {{
    "top_pick": "company name",
    "reasoning": "why this company is preferred",
    "concerns": "what to watch"
  }}
}}

Return ONLY the JSON object."""
        
        result = self.rag.query(prompt)
        
        try:
            json_start = result['answer'].find('{')
            json_end = result['answer'].rfind('}') + 1
            json_str = result['answer'][json_start:json_end]
            comparison = json.loads(json_str)
            return comparison
        except:
            return {"error": "Parsing failed", "raw": result['answer']}
    
    def generate_investment_thesis(self, company: str, investment_horizon: str = "long-term") -> Dict:
        """
        Generate bull and bear cases for investment
        
        USE CASE: Investment memos, pitch decks, portfolio reviews
        
        ═══════════════════════════════════════════════════════════════
        ARGUMENT GENERATION
        ═══════════════════════════════════════════════════════════════
        
        LLMs excel at:
        - Synthesizing multiple data points into coherent arguments
        - Balancing perspectives (bull vs bear cases)
        - Connecting evidence to conclusions
        
        PROMPT DESIGN INSIGHT:
        - Ask for both sides (reduces bias)
        - Require evidence citations (reduces hallucinations)
        - Specify investment horizon (different factors matter)
        """
        
        prompt = f"""Develop a comprehensive investment thesis for {company} 
with a {investment_horizon} investment horizon.

Provide both bull and bear cases:

Return as JSON:
{{
  "company": "{company}",
  "investment_horizon": "{investment_horizon}",
  "bull_case": {{
    "thesis_summary": "one paragraph bull case",
    "key_drivers": ["driver 1", "driver 2", "driver 3"],
    "upside_catalysts": ["catalyst 1", "catalyst 2"],
    "potential_return": "estimated upside percentage or range"
  }},
  "bear_case": {{
    "thesis_summary": "one paragraph bear case",
    "key_risks": ["risk 1", "risk 2", "risk 3"],
    "downside_triggers": ["trigger 1", "trigger 2"],
    "potential_loss": "estimated downside percentage or range"
  }},
  "base_case": {{
    "most_likely_scenario": "balanced outcome description",
    "key_assumptions": ["assumption 1", "assumption 2"],
    "recommended_action": "Buy/Hold/Sell"
  }},
  "key_metrics_to_monitor": ["metric 1", "metric 2", "metric 3"]
}}

Base analysis on financial documents. Cite specific data points.
Return ONLY the JSON object."""
        
        result = self.rag.query(prompt)
        
        try:
            json_start = result['answer'].find('{')
            json_end = result['answer'].rfind('}') + 1
            json_str = result['answer'][json_start:json_end]
            thesis = json.loads(json_str)
            thesis['generated_at'] = datetime.now().isoformat()
            return thesis
        except:
            return {"error": "Parsing failed", "raw": result['answer']}
    
    def extract_key_metrics(self, company: str) -> pd.DataFrame:
        """
        Extract structured financial metrics into DataFrame
        
        USE CASE: Quantitative analysis, screening, modeling
        
        ═══════════════════════════════════════════════════════════════
        STRUCTURED DATA EXTRACTION
        ═══════════════════════════════════════════════════════════════
        
        Pattern: LLM extracts → JSON → DataFrame
        Why? Financial models need tabular data
        
        PRODUCTION CONSIDERATION:
        - LLMs can make calculation errors
        - Always validate critical numbers
        - Consider hybrid: LLM extracts, deterministic code calculates
        
        INTERVIEW TIP: Discuss when to use LLM vs traditional code
        - LLM: Unstructured text → structured data
        - Code: Calculations, transformations, business logic
        """
        
        prompt = f"""Extract key financial metrics for {company} from the documents.

Return as JSON array of metrics:
[
  {{
    "metric": "Revenue",
    "current_period": {{"value": 100000, "period": "Q3 2024"}},
    "prior_period": {{"value": 95000, "period": "Q3 2023"}},
    "yoy_change_pct": 5.3
  }},
  {{
    "metric": "Net Income",
    "current_period": {{"value": 15000, "period": "Q3 2024"}},
    "prior_period": {{"value": 14000, "period": "Q3 2023"}},
    "yoy_change_pct": 7.1
  }}
]

Extract these metrics if available:
- Revenue (total and by segment if disclosed)
- Net Income
- Operating Income
- Gross Margin %
- Operating Margin %
- EPS (earnings per share)
- Free Cash Flow
- R&D Spending
- CapEx
- Debt/Equity Ratio

Return ONLY the JSON array."""
        
        result = self.rag.query(prompt)
        
        try:
            json_start = result['answer'].find('[')
            json_end = result['answer'].rfind(']') + 1
            json_str = result['answer'][json_start:json_end]
            metrics = json.loads(json_str)
            df = pd.DataFrame(metrics)
            return df
        except:
            return pd.DataFrame({"error": ["Failed to extract metrics"]})


# ============================================================================
# USAGE EXAMPLE & EVALUATION
# ============================================================================

def main():
    """
    Demonstrate financial analysis workflow
    
    PRODUCTION WORKFLOW:
    1. Collect data (Module 1)
    2. Build RAG index (Module 2)
    3. Run analysis (Module 3) ← We are here
    4. Present results (Module 4)
    """
    
    from src.rag_engine.financial_rag import FinancialRAGEngine
    
    print("\n" + "="*60)
    print("📊 FINANCIAL ANALYSIS DEMO")
    print("="*60 + "\n")
    
    # Initialize RAG engine
    rag = FinancialRAGEngine()
    rag.load_existing_index("data/chroma_db")
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer(rag)
    
    company = "AAPL"
    
    # Run different analyses
    print(f"\n{'─'*60}")
    print("1️⃣  EXECUTIVE SUMMARY")
    print(f"{'─'*60}\n")
    summary = analyzer.generate_executive_summary(company)
    print(json.dumps(summary, indent=2))
    
    print(f"\n{'─'*60}")
    print("2️⃣  RISK ANALYSIS")
    print(f"{'─'*60}\n")
    risks = analyzer.analyze_risk_factors(company)
    print(json.dumps(risks, indent=2))
    
    print(f"\n{'─'*60}")
    print("3️⃣  INVESTMENT THESIS")
    print(f"{'─'*60}\n")
    thesis = analyzer.generate_investment_thesis(company, "long-term")
    print(json.dumps(thesis, indent=2))
    
    print(f"\n{'─'*60}")
    print("4️⃣  KEY METRICS")
    print(f"{'─'*60}\n")
    metrics_df = analyzer.extract_key_metrics(company)
    print(metrics_df.to_string())
    
    # Save results
    results = {
        "company": company,
        "analysis_date": datetime.now().isoformat(),
        "executive_summary": summary,
        "risk_analysis": risks,
        "investment_thesis": thesis,
        "key_metrics": metrics_df.to_dict('records') if not metrics_df.empty else {}
    }
    
    output_file = f"data/processed/{company}_analysis_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Full analysis saved to {output_file}")


if __name__ == "__main__":
    main()


"""
=============================================================================
INTERVIEW PREPARATION - PRODUCTION AI SYSTEMS
=============================================================================

🎯 KEY DISCUSSION POINTS:

1. STRUCTURED OUTPUTS
   "I designed the system to return JSON for easy integration with 
   downstream systems. This enables automated workflows like populating 
   dashboards or triggering alerts."

2. ERROR HANDLING
   "Production systems need graceful degradation. If JSON parsing fails,
   I return the raw response rather than crashing. This allows manual 
   review and debugging."

3. PROMPT ENGINEERING PROCESS
   "I iterate on prompts by:
   - Testing with sample inputs
   - Reviewing outputs for quality
   - Adjusting specificity and constraints
   - Measuring against success criteria"

4. COST OPTIMIZATION
   "Each analysis call costs ~$0.05-0.20 depending on document size.
   Optimization strategies:
   - Cache frequent queries
   - Use cheaper models for simple tasks
   - Batch similar queries
   - Implement usage quotas"

5. QUALITY ASSURANCE
   "I evaluate outputs using:
   - Spot-checking against source documents
   - User feedback loops
   - Comparison against baseline (e.g., analyst reports)
   - Monitoring for hallucinations/errors"

ADVANCED INTERVIEW QUESTIONS:

Q: "How do you ensure consistency across runs?"
A: "Temperature=0 for deterministic outputs. For critical applications,
   run multiple times and aggregate or flag disagreements."

Q: "What if the LLM refuses to answer?"
A: "Implement retry logic with adjusted prompts. Log refusals for 
   analysis. Consider model fine-tuning if systematic."

Q: "How would you version control prompts?"
A: "Store prompts in config files or database. Track changes with Git.
   A/B test prompt versions in production to measure impact."

Q: "What about data privacy for sensitive financial information?"
A: "Options:
   - Use on-premise LLMs (e.g., Llama 2)
   - Azure OpenAI (dedicated instances)
   - Anonymize data before processing
   - Implement data retention policies"

METRICS TO DISCUSS:

- Latency: P50, P95, P99 response times
- Cost: $ per query, $ per user per month
- Quality: Accuracy %, hallucination rate, user satisfaction
- Reliability: Uptime, error rate, retry success rate

NEXT MODULE:
Build a Streamlit dashboard that brings everything together into
a production-ready user interface.
=============================================================================
"""