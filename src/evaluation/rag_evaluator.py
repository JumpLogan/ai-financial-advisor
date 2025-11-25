"""
STEP 2: RAG Evaluation Module
File: src/evaluation/rag_evaluator.py

PURPOSE: Systematic testing and evaluation of RAG system

METRICS COVERED:
1. Retrieval Quality (precision@k, recall@k, MRR)
2. Answer Quality (relevance, faithfulness, correctness)
3. Hallucination Detection
4. Performance (latency P50/P95/P99, cost, throughput)

USAGE:
    from src.evaluation.rag_evaluator import RAGEvaluator, TestCase
    
    evaluator = RAGEvaluator(rag_engine)
    test_cases = [TestCase(question="...", expected_topics=[...])]
    results = evaluator.run_evaluation(test_cases)
    report = evaluator.generate_report(results)
"""

import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import os


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    """Single test case for evaluation"""
    question: str
    expected_topics: List[str] = None  # Topics that should be mentioned
    category: str = "general"  # revenue, risk, strategy, etc.
    expected_answer: Optional[str] = None  # Ground truth (if available)
    company: Optional[str] = None
    
    def __post_init__(self):
        if self.expected_topics is None:
            self.expected_topics = []


@dataclass
class EvaluationResult:
    """Results from evaluating a single test case"""
    question: str
    answer: str
    retrieved_docs: List[str]
    
    # Quality metrics
    relevance_score: float = 0.0  # Did answer address question?
    faithfulness_score: float = 0.0  # Is answer grounded in docs?
    topic_coverage: float = 0.0  # Percentage of expected topics covered
    
    # Performance metrics
    latency: float = 0.0
    num_tokens: int = 0
    cost_estimate: float = 0.0
    
    # Flags
    has_hallucination: bool = False
    is_acceptable: bool = False
    
    # Metadata
    timestamp: str = ""
    category: str = "general"


# ═══════════════════════════════════════════════════════════════════════════
# METRICS CALCULATORS
# ═══════════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """Calculate various quality metrics"""
    
    @staticmethod
    def calculate_relevance(question: str, answer: str) -> float:
        """
        Simple relevance score based on word overlap
        
        More sophisticated: Use sentence-transformers for semantic similarity
        """
        if not answer:
            return 0.0
        
        # Tokenize and lowercase
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'what', 'how', 'when', 'where', 'why', 'who'}
        question_words = question_words - stop_words
        answer_words = answer_words - stop_words
        
        if not question_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(question_words.intersection(answer_words))
        return min(overlap / len(question_words), 1.0)
    
    @staticmethod
    def calculate_faithfulness(answer: str, context_docs: List[str]) -> float:
        """
        Check if answer is grounded in retrieved documents
        
        Method: Check what fraction of answer content appears in context
        """
        if not answer or not context_docs:
            return 0.0
        
        context_text = ' '.join(context_docs).lower()
        
        # Split answer into sentences
        sentences = answer.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return 0.5
        
        grounded_count = 0
        
        for sentence in sentences:
            # Check if key content words from sentence appear in context
            words = [w for w in sentence.lower().split() if len(w) > 4]
            if not words:
                continue
            
            # Calculate how many words appear in context
            found = sum(1 for w in words if w in context_text)
            
            # If >50% of content words found, consider sentence grounded
            if found / len(words) > 0.5:
                grounded_count += 1
        
        return grounded_count / len(sentences) if sentences else 0.0
    
    @staticmethod
    def calculate_topic_coverage(answer: str, expected_topics: List[str]) -> Tuple[float, List[str]]:
        """
        Check if answer covers expected topics
        
        Returns: (coverage_score, missing_topics)
        """
        if not expected_topics:
            return 1.0, []
        
        answer_lower = answer.lower()
        covered = []
        missing = []
        
        for topic in expected_topics:
            if topic.lower() in answer_lower:
                covered.append(topic)
            else:
                missing.append(topic)
        
        coverage = len(covered) / len(expected_topics)
        return coverage, missing
    
    @staticmethod
    def detect_hallucination(answer: str, context_docs: List[str]) -> Tuple[bool, List[str]]:
        """
        Detect potential hallucinations (unsupported numeric claims)
        
        Returns: (has_hallucination, suspicious_claims)
        """
        import re
        
        context_text = ' '.join(context_docs).lower()
        suspicious = []
        
        # Find numeric claims in answer
        sentences = answer.split('.')
        
        for sentence in sentences:
            # Look for numbers or percentages
            numbers = re.findall(r'\d+\.?\d*\s*%?', sentence)
            
            if numbers:
                # Check if ANY of these numbers appear in context
                found = any(num in context_text for num in numbers)
                
                if not found and len(sentence.strip()) > 20:
                    suspicious.append(sentence.strip())
        
        has_hallucination = len(suspicious) > 0
        return has_hallucination, suspicious
    
    @staticmethod
    def estimate_cost(text: str, model: str = "gpt-3.5-turbo") -> float:
        """
        Estimate API cost for a query
        
        Rough token estimation: 1 token ≈ 0.75 words
        """
        word_count = len(text.split())
        token_estimate = word_count / 0.75
        
        # Pricing (per 1K tokens)
        prices = {
            "gpt-3.5-turbo": 0.0015,  # $0.0015 per 1K tokens (average of input/output)
            "gpt-4": 0.045,  # $0.045 per 1K tokens (average)
        }
        
        price_per_1k = prices.get(model, 0.002)
        return (token_estimate / 1000) * price_per_1k


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    """
    Comprehensive RAG system evaluator
    
    USAGE:
        evaluator = RAGEvaluator(rag_engine)
        results = evaluator.run_evaluation(test_cases)
        evaluator.generate_report(results)
    """
    
    def __init__(self, rag_engine, model_name: str = "gpt-3.5-turbo"):
        self.rag_engine = rag_engine
        self.model_name = model_name
        self.metrics = MetricsCalculator()
    
    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        
        start_time = time.time()
        
        try:
            # Query RAG system
            result = self.rag_engine.query(test_case.question, return_sources=True)
            
            answer = result['answer']
            sources = result.get('sources', [])
            retrieved_docs = [s.get('excerpt', s.get('page_content', '')) for s in sources]
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return EvaluationResult(
                question=test_case.question,
                answer=f"ERROR: {e}",
                retrieved_docs=[],
                latency=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                category=test_case.category
            )
        
        latency = time.time() - start_time
        
        # Calculate metrics
        relevance = self.metrics.calculate_relevance(test_case.question, answer)
        faithfulness = self.metrics.calculate_faithfulness(answer, retrieved_docs)
        topic_coverage, missing_topics = self.metrics.calculate_topic_coverage(
            answer, test_case.expected_topics
        )
        has_hallucination, suspicious = self.metrics.detect_hallucination(answer, retrieved_docs)
        
        # Cost estimation
        total_text = test_case.question + answer + ' '.join(retrieved_docs)
        cost = self.metrics.estimate_cost(total_text, self.model_name)
        num_tokens = int(len(total_text.split()) / 0.75)
        
        # Determine if acceptable
        is_acceptable = (
            relevance > 0.5 and
            faithfulness > 0.6 and
            not has_hallucination
        )
        
        result = EvaluationResult(
            question=test_case.question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            relevance_score=relevance,
            faithfulness_score=faithfulness,
            topic_coverage=topic_coverage,
            latency=latency,
            num_tokens=num_tokens,
            cost_estimate=cost,
            has_hallucination=has_hallucination,
            is_acceptable=is_acceptable,
            timestamp=datetime.now().isoformat(),
            category=test_case.category
        )
        
        # Print summary
        status = "✅" if is_acceptable else "⚠️"
        print(f"   {status} Relevance: {relevance:.2f} | Faithfulness: {faithfulness:.2f} | Latency: {latency:.2f}s")
        
        return result
    
    def run_evaluation(self, test_cases: List[TestCase]) -> List[EvaluationResult]:
        """Run evaluation on multiple test cases"""
        
        print(f"\n{'='*70}")
        print(f"🧪 EVALUATING {len(test_cases)} TEST CASES")
        print(f"{'='*70}\n")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] {test_case.question[:60]}...")
            result = self.evaluate_single(test_case)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> str:
        """Generate comprehensive evaluation report"""
        
        if not results:
            return "No results to report"
        
        # Calculate aggregates
        df = pd.DataFrame([asdict(r) for r in results])
        
        total = len(results)
        passed = sum(1 for r in results if r.is_acceptable)
        failed = total - passed
        hallucinations = sum(1 for r in results if r.has_hallucination)
        
        avg_relevance = df['relevance_score'].mean()
        avg_faithfulness = df['faithfulness_score'].mean()
        avg_topic_coverage = df['topic_coverage'].mean()
        
        avg_latency = df['latency'].mean()
        p50_latency = df['latency'].quantile(0.5)
        p95_latency = df['latency'].quantile(0.95)
        p99_latency = df['latency'].quantile(0.99)
        
        total_cost = df['cost_estimate'].sum()
        cost_per_query = total_cost / total
        
        # Build report
        report = f"""
# RAG System Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** {self.model_name}  
**Test Cases:** {total}

---

## Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Tests Passed** | {passed}/{total} ({passed/total*100:.1f}%) | {'✅' if passed/total > 0.8 else '⚠️'} |
| **Tests Failed** | {failed}/{total} ({failed/total*100:.1f}%) | {'✅' if failed/total < 0.2 else '⚠️'} |
| **Hallucinations** | {hallucinations} | {'✅' if hallucinations == 0 else '⚠️'} |

---

## Quality Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Answer Relevance | {avg_relevance:.3f} | >0.70 | {'✅' if avg_relevance > 0.7 else '⚠️'} |
| Faithfulness | {avg_faithfulness:.3f} | >0.70 | {'✅' if avg_faithfulness > 0.7 else '⚠️'} |
| Topic Coverage | {avg_topic_coverage:.3f} | >0.60 | {'✅' if avg_topic_coverage > 0.6 else '⚠️'} |

---

## Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Avg Latency** | {avg_latency:.2f}s | <2.0s | {'✅' if avg_latency < 2.0 else '⚠️'} |
| **P50 Latency** | {p50_latency:.2f}s | <1.5s | {'✅' if p50_latency < 1.5 else '⚠️'} |
| **P95 Latency** | {p95_latency:.2f}s | <5.0s | {'✅' if p95_latency < 5.0 else '⚠️'} |
| **P99 Latency** | {p99_latency:.2f}s | <10.0s | {'✅' if p99_latency < 10.0 else '⚠️'} |
| **Total Cost** | ${total_cost:.4f} | - | - |
| **Cost per Query** | ${cost_per_query:.4f} | <$0.05 | {'✅' if cost_per_query < 0.05 else '⚠️'} |

---

## Detailed Results by Category

"""
        
        # Results by category
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            cat_passed = (cat_df['is_acceptable']).sum()
            cat_total = len(cat_df)
            
            report += f"### {category.title()}\n"
            report += f"- Tests: {cat_total}\n"
            report += f"- Passed: {cat_passed}/{cat_total} ({cat_passed/cat_total*100:.0f}%)\n"
            report += f"- Avg Relevance: {cat_df['relevance_score'].mean():.2f}\n"
            report += f"- Avg Faithfulness: {cat_df['faithfulness_score'].mean():.2f}\n\n"
        
        # Recommendations
        report += "---\n\n## Recommendations\n\n"
        
        if avg_faithfulness < 0.7:
            report += "⚠️ **High Hallucination Risk**\n"
            report += "- Use more specific prompts with 'cite sources' instruction\n"
            report += "- Increase number of retrieved documents (k)\n"
            report += "- Consider using temperature=0.0 for factual queries\n\n"
        
        if avg_latency > 2.0:
            report += "⚠️ **High Latency**\n"
            report += "- Consider using gpt-3.5-turbo instead of gpt-4\n"
            report += "- Reduce chunk size for faster retrieval\n"
            report += "- Implement caching for common queries\n\n"
        
        if cost_per_query > 0.05:
            report += "⚠️ **High Cost per Query**\n"
            report += "- Use cheaper model (gpt-3.5-turbo)\n"
            report += "- Reduce number of retrieved chunks\n"
            report += "- Cache expensive operations\n\n"
        
        if avg_relevance < 0.6:
            report += "⚠️ **Low Answer Relevance**\n"
            report += "- Improve prompt engineering\n"
            report += "- Adjust retrieval strategy\n"
            report += "- Consider fine-tuning embeddings\n\n"
        
        if passed / total > 0.9:
            report += "✅ **System Performing Well**\n"
            report += "- Maintain current configuration\n"
            report += "- Monitor for regressions\n"
            report += "- Consider A/B testing optimizations\n\n"
        
        # Failed cases
        failed_results = [r for r in results if not r.is_acceptable]
        if failed_results:
            report += "---\n\n## Failed Test Cases\n\n"
            report += "| Question | Issue | Relevance | Faithfulness |\n"
            report += "|----------|-------|-----------|-------------|\n"
            
            for r in failed_results[:10]:  # Show first 10
                issue = "Hallucination" if r.has_hallucination else "Low Quality"
                q_short = r.question[:50] + "..." if len(r.question) > 50 else r.question
                report += f"| {q_short} | {issue} | {r.relevance_score:.2f} | {r.faithfulness_score:.2f} |\n"
        
        return report
    
    def save_results(self, results: List[EvaluationResult], filename: str = "evaluation_results.json"):
        """Save results to JSON"""
        results_dict = [asdict(r) for r in results]
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
    
    def save_report(self, report: str, filename: str = "evaluation_report.md"):
        """Save report to markdown file"""
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to {filename}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST CASE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def create_example_test_cases() -> List[TestCase]:
    """Create example test cases for financial RAG"""
    
    return [
        TestCase(
            question="What was the revenue growth rate in the most recent period?",
            expected_topics=["revenue", "growth", "percent", "year"],
            category="revenue"
        ),
        TestCase(
            question="What are the main risk factors for the company?",
            expected_topics=["risk", "factors", "challenges"],
            category="risk"
        ),
        TestCase(
            question="How much did the company spend on R&D?",
            expected_topics=["R&D", "research", "development", "spending"],
            category="expense"
        ),
        TestCase(
            question="What is the company's competitive advantage?",
            expected_topics=["competitive", "advantage", "differentiation"],
            category="strategy"
        ),
        TestCase(
            question="What was the net income?",
            expected_topics=["net income", "profit", "earnings"],
            category="profitability"
        ),
        TestCase(
            question="How has the company's debt changed?",
            expected_topics=["debt", "liability", "borrowing"],
            category="financial_health"
        ),
        TestCase(
            question="What are the key strategic initiatives?",
            expected_topics=["strategy", "initiative", "priority", "focus"],
            category="strategy"
        ),
        TestCase(
            question="What is the cash position?",
            expected_topics=["cash", "liquidity", "cash equivalents"],
            category="financial_health"
        ),
    ]


def load_test_cases_from_json(filepath: str) -> List[TestCase]:
    """Load test cases from JSON file"""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    test_cases = []
    for item in data.get('test_cases', []):
        test_cases.append(TestCase(
            question=item['question'],
            expected_topics=item.get('expected_topics', []),
            category=item.get('category', 'general'),
            expected_answer=item.get('expected_answer'),
            company=item.get('company')
        ))
    
    return test_cases


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Example: Run evaluation"""
    
    import sys
    sys.path.append('../../')
    from rag_engine.financial_rag import FinancialRAGEngine
    
    print("\n" + "="*70)
    print("🧪 RAG SYSTEM EVALUATION")
    print("="*70)
    
    # Load RAG system
    print("\n📚 Loading RAG system...")
    rag = FinancialRAGEngine()
    rag.load_existing_index("../../data/chroma_db")
    print("✅ RAG system loaded")
    
    # Create or load test cases
    print("\n📝 Loading test cases...")
    test_cases = create_example_test_cases()
    print(f"✅ Loaded {len(test_cases)} test cases")
    
    # Run evaluation
    evaluator = RAGEvaluator(rag)
    results = evaluator.run_evaluation(test_cases)
    
    # Generate and save report
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    evaluator.save_report(report, "evaluation_report.md")
    evaluator.save_results(results, "evaluation_results.json")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()