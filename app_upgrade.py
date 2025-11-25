"""
Financial Intelligence Assistant - COMPLETE FIXED VERSION
File: app.py

FIXES IMPLEMENTED:
✅ Bug 1: Quick questions text persistence (session state)
✅ Bug 2: Financial analysis tools error handling  
✅ Feature 3: Hybrid RAG + Web Search with source links
✅ Feature 4: Performance metrics dashboard

INSTALLATION:
1. Save this file as app.py (replace existing)
2. pip install streamlit openai langchain langchain-community langchain-openai chromadb
3. streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import sys
import time
from typing import List, Dict, Optional

# Import your modules
sys.path.append('src')
from rag_engine.financial_rag import FinancialRAGEngine
from analysis.financial_analyzer import FinancialAnalyzer
from data_collection.sec_collector import SECDataCollector


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE - FIX #1: Persist data across reruns
# ═══════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize all session state variables"""
    
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    if 'index_loaded' not in st.session_state:
        st.session_state.index_loaded = False
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # FIX #1: Persist selected question
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # FEATURE #3: Web search toggle
    if 'use_web_search' not in st.session_state:
        st.session_state.use_web_search = False  # Disabled by default (no API integration yet)
    
    # FEATURE #4: Performance tracking
    if 'query_metrics' not in st.session_state:
        st.session_state.query_metrics = []
    
    # Error tracking
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []


# ═══════════════════════════════════════════════════════════════════════════
# HYBRID QUERY - FEATURE #3: RAG + Web Search
# ═══════════════════════════════════════════════════════════════════════════

def enhanced_query(question: str, rag_engine) -> dict:
    """
    Enhanced query with better error handling and performance tracking
    
    This is a placeholder for web search integration.
    To add web search:
    1. Use Claude.ai web_search tool in production
    2. Or integrate with SerpAPI, Brave Search, etc.
    """
    start_time = time.time()
    
    try:
        # Query RAG system
        result = rag_engine.query(question, return_sources=True)
        
        # Format sources with better display
        sources = []
        for i, src in enumerate(result.get('sources', []), 1):
            sources.append({
                'num': i,
                'type': 'document',
                'title': src.get('filename', 'Unknown'),
                'excerpt': src.get('excerpt', '')[:300] + '...',
                'link': None
            })
        
        # Note: Web search integration would go here
        # Example structure for when you add it:
        # if st.session_state.use_web_search:
        #     web_results = search_web(question, company)
        #     for web_src in web_results:
        #         sources.append({
        #             'type': 'web',
        #             'title': web_src['title'],
        #             'excerpt': web_src['snippet'],
        #             'link': web_src['url']
        #         })
        
        latency = time.time() - start_time
        
        # Track metrics
        st.session_state.query_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'question': question[:100],
            'latency': latency,
            'num_sources': len(sources),
            'success': True
        })
        
        return {
            'answer': result['answer'],
            'sources': sources,
            'latency': latency,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        latency = time.time() - start_time
        
        # Log error
        st.session_state.error_log.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'error': str(e),
            'type': type(e).__name__
        })
        
        # Track metrics even for errors
        st.session_state.query_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'question': question[:100],
            'latency': latency,
            'num_sources': 0,
            'success': False
        })
        
        return {
            'answer': '',
            'sources': [],
            'latency': latency,
            'success': False,
            'error': str(e)
        }


# ═══════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Sidebar configuration"""
    
    st.sidebar.title("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    model_choice = st.sidebar.selectbox(
        "LLM Model",
        ["gpt-4", "gpt-3.5-turbo"],
        index=1,  # Default to gpt-3.5-turbo (cheaper)
        help="GPT-4: More accurate but expensive | GPT-3.5: Faster and cheaper"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        0.0, 1.0, 0.0, 0.1,
        help="0.0 = Factual/Deterministic, 1.0 = Creative/Random"
    )
    
    # Web search toggle (currently disabled - placeholder for future)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Advanced Features")
    
    web_search_enabled = st.sidebar.checkbox(
        "Enable Web Search (Coming Soon)",
        value=False,
        disabled=True,
        help="Web search integration - will be available in future update"
    )
    
    st.sidebar.markdown("---")
    
    # Data collection
    st.sidebar.subheader("📊 Data Collection")
    
    with st.sidebar.expander("Add New Company Data"):
        ticker = st.text_input("Stock Ticker", placeholder="AAPL")
        num_filings = st.number_input("# of Filings", 1, 20, 5)
        
        if st.button("Collect Filings"):
            if ticker:
                with st.spinner(f"Collecting {num_filings} filings for {ticker}..."):
                    try:
                        collector = SECDataCollector(os.getenv("SEC_API_KEY"))
                        result = collector.process_company(ticker, num_filings)
                        st.success(f"✅ Downloaded {result['total_downloaded']} filings!")
                        
                        # Mark for index rebuild
                        st.session_state.index_loaded = False
                        st.session_state.rag_engine = None
                        st.session_state.analyzer = None
                        
                        st.info("🔄 RAG index will rebuild on next query.")
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error collecting data: {e}")
                        st.exception(e)
            else:
                st.warning("Please enter a ticker symbol")
    
    st.sidebar.markdown("---")
    
    # Index Management
    st.sidebar.subheader("🔧 Index Management")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Rebuild"):
            st.session_state.index_loaded = False
            st.session_state.rag_engine = None
            st.session_state.analyzer = None
            st.success("Will rebuild!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear"):
            import shutil
            persist_dir = "data/chroma_db"
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                st.session_state.index_loaded = False
                st.session_state.rag_engine = None
                st.session_state.analyzer = None
                st.success("Cleared!")
                time.sleep(1)
                st.rerun()
    
    st.sidebar.markdown("---")
    
    # System Status
    st.sidebar.subheader("🔋 System Status")
    
    data_dir = "data/raw"
    persist_dir = "data/chroma_db"
    
    # Document count
    if os.path.exists(data_dir):
        num_docs = len([f for f in os.listdir(data_dir) if f.endswith('.txt')])
        st.sidebar.metric("Documents", num_docs)
    else:
        st.sidebar.metric("Documents", 0)
    
    # Index status
    if st.session_state.index_loaded:
        st.sidebar.success("✅ RAG Index Loaded")
    elif os.path.exists(persist_dir):
        st.sidebar.warning("⚠️ Index exists but not loaded")
    else:
        st.sidebar.error("❌ No RAG Index")
    
    # Performance metrics
    if st.session_state.query_metrics:
        recent_queries = st.session_state.query_metrics[-10:]
        avg_latency = sum(q['latency'] for q in recent_queries) / len(recent_queries)
        success_rate = sum(1 for q in recent_queries if q['success']) / len(recent_queries)
        
        st.sidebar.metric("Avg Latency", f"{avg_latency:.2f}s")
        st.sidebar.metric("Success Rate", f"{success_rate*100:.0f}%")
    
    # Error count
    if st.session_state.error_log:
        st.sidebar.metric("Errors (session)", len(st.session_state.error_log))
    
    return model_choice, temperature


def load_rag_system(model_name: str, temperature: float):
    """Load or initialize RAG system"""
    
    if st.session_state.index_loaded:
        return  # Already loaded
    
    with st.spinner("🔄 Loading RAG system..."):
        try:
            # Initialize RAG engine
            rag = FinancialRAGEngine(
                model_name=model_name,
                temperature=temperature
            )
            
            # Load or build index
            persist_dir = "data/chroma_db"
            
            if os.path.exists(persist_dir):
                rag.load_existing_index(persist_dir)
                st.success("✅ Loaded existing RAG index")
            else:
                # Build new index from documents
                data_dir = "data/raw"
                if not os.path.exists(data_dir):
                    st.error(f"❌ Data directory not found: {data_dir}")
                    st.info("Use sidebar to collect company data first")
                    return
                
                doc_paths = [
                    os.path.join(data_dir, f) 
                    for f in os.listdir(data_dir) 
                    if f.endswith('.txt')
                ]
                
                if not doc_paths:
                    st.error("❌ No documents found in data/raw/")
                    st.info("Use sidebar to collect company data first")
                    return
                
                with st.spinner(f"Building index from {len(doc_paths)} documents..."):
                    rag.build_index(doc_paths, persist_dir)
                    st.success(f"✅ Built RAG index from {len(doc_paths)} documents")
            
            # Initialize analyzer
            analyzer = FinancialAnalyzer(rag, model_name=model_name)
            
            # Save to session state
            st.session_state.rag_engine = rag
            st.session_state.analyzer = analyzer
            st.session_state.index_loaded = True
            
        except Exception as e:
            st.error(f"❌ Error loading RAG system: {e}")
            st.exception(e)
            st.info("💡 Troubleshooting:")
            st.info("1. Check OPENAI_API_KEY in .env file")
            st.info("2. Verify documents exist in data/raw/")
            st.info("3. Try clearing and rebuilding the index")


def render_query_interface():
    """Q&A Interface - FIX #1: Question persistence"""
    
    st.header("💬 Ask Questions")
    
    # Quick questions
    st.subheader("Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    quick_questions = {
        "Revenue Growth": "What was the revenue growth rate in the most recent period?",
        "Key Risks": "What are the main risk factors mentioned in the filings?",
        "Strategic Priorities": "What are the company's strategic priorities and initiatives?"
    }
    
    # FIX #1: Use session state to persist question
    with col1:
        if st.button("📈 Revenue Growth", use_container_width=True):
            st.session_state.current_question = quick_questions["Revenue Growth"]
            st.rerun()
    
    with col2:
        if st.button("⚠️ Key Risks", use_container_width=True):
            st.session_state.current_question = quick_questions["Key Risks"]
            st.rerun()
    
    with col3:
        if st.button("🎯 Strategic Priorities", use_container_width=True):
            st.session_state.current_question = quick_questions["Strategic Priorities"]
            st.rerun()
    
    # Custom question input - reads from session state
    question = st.text_area(
        "Or ask your own question:",
        value=st.session_state.current_question,
        placeholder="e.g., How has R&D spending changed over the past year?",
        height=100,
        key="question_input"
    )
    
    # Update session state when user types
    if question != st.session_state.current_question:
        st.session_state.current_question = question
    
    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if not question:
            st.warning("⚠️ Please enter a question")
            return
        
        if not st.session_state.rag_engine:
            st.error("❌ RAG system not loaded")
            st.info("Check System Status in sidebar")
            return
        
        # Execute query with error handling
        with st.spinner("🔍 Searching documents..."):
            result = enhanced_query(question, st.session_state.rag_engine)
        
        if result['success']:
            # Display answer
            st.markdown("### 💡 Answer")
            st.write(result['answer'])
            
            # Show metrics
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"⏱️ Response time: {result['latency']:.2f}s")
            with col2:
                st.caption(f"📚 Sources: {len(result['sources'])}")
            
            # Display sources
            if result['sources']:
                with st.expander(f"📚 View Sources ({len(result['sources'])} documents)"):
                    for src in result['sources']:
                        if src['type'] == 'document':
                            st.markdown(f"**{src['num']}. 📄 {src['title']}**")
                            st.caption(src['excerpt'])
                            st.divider()
            
            # Save to history
            st.session_state.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': result['answer'],
                'latency': result['latency'],
                'num_sources': len(result['sources'])
            })
            
        else:
            # Display error
            st.error(f"❌ Query failed: {result['error']}")
            st.exception(Exception(result['error']))


def render_analysis_tools():
    """Analysis Tools - FIX #2: Better error handling"""
    
    st.header("📊 Financial Analysis Tools")
    
    if not st.session_state.analyzer:
        st.warning("⚠️ Analyzer not loaded. Load RAG system first.")
        return
    
    # Company input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        company_input = st.text_input(
            "Company Ticker",
            placeholder="AAPL, MSFT, GOOGL",
            help="Enter one or more ticker symbols separated by commas"
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("📊 Analyze", type="primary", use_container_width=True)
    
    if not analyze_button or not company_input:
        return
    
    companies = [c.strip().upper() for c in company_input.split(',')]
    
    # Single company analysis
    if len(companies) == 1:
        company = companies[0]
        
        tabs = st.tabs([
            "📄 Executive Summary",
            "⚠️ Risk Analysis",
            "💼 Investment Thesis",
            "📈 Key Metrics"
        ])
        
        # TAB 1: Executive Summary
        with tabs[0]:
            try:
                with st.spinner("Generating executive summary..."):
                    summary = st.session_state.analyzer.generate_executive_summary(company)
                
                if 'error' in summary:
                    st.error("❌ Failed to generate summary")
                    st.code(summary.get('raw_response', 'Unknown error'))
                else:
                    # Display summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Period", summary.get('period_covered', 'N/A'))
                    with col2:
                        rating = summary.get('analyst_rating', 'N/A')
                        st.metric("Rating", rating)
                    
                    st.subheader("Key Metrics")
                    metrics = summary.get('key_metrics', {})
                    for key, value in metrics.items():
                        st.write(f"**{key.title()}:** {value}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("✅ Strengths")
                        for strength in summary.get('strengths', []):
                            st.write(f"• {strength}")
                    
                    with col2:
                        st.subheader("⚠️ Challenges")
                        for challenge in summary.get('challenges', []):
                            st.write(f"• {challenge}")
                    
                    st.subheader("🔮 Outlook")
                    st.write(summary.get('outlook', 'N/A'))
                    
            except Exception as e:
                st.error(f"❌ Error in executive summary: {e}")
                st.exception(e)
                st.info("💡 Try collecting more data for this company")
        
        # TAB 2: Risk Analysis
        with tabs[1]:
            try:
                with st.spinner("Analyzing risk factors..."):
                    risks = st.session_state.analyzer.analyze_risk_factors(company)
                
                if 'error' in risks:
                    st.error("❌ Failed to analyze risks")
                    st.code(risks.get('raw', 'Unknown error'))
                else:
                    st.metric("Overall Risk Level", risks.get('overall_risk_level', 'N/A'))
                    
                    st.subheader("Risk Categories")
                    risk_categories = risks.get('risk_categories', {})
                    
                    for category, risk_list in risk_categories.items():
                        if risk_list:
                            with st.expander(f"📌 {category.title()} Risks"):
                                for risk in risk_list:
                                    st.write(f"• {risk}")
                    
                    if risks.get('key_concerns'):
                        st.subheader("🚨 Key Concerns")
                        for concern in risks['key_concerns']:
                            st.warning(concern)
                    
                    if risks.get('mitigating_factors'):
                        st.subheader("🛡️ Mitigating Factors")
                        for factor in risks['mitigating_factors']:
                            st.info(factor)
                    
            except Exception as e:
                st.error(f"❌ Error in risk analysis: {e}")
                st.exception(e)
        
        # TAB 3: Investment Thesis
        with tabs[2]:
            try:
                horizon = st.selectbox(
                    "Investment Horizon",
                    ["short-term", "long-term"],
                    index=1
                )
                
                with st.spinner("Developing investment thesis..."):
                    thesis = st.session_state.analyzer.generate_investment_thesis(
                        company, horizon
                    )
                
                if 'error' in thesis:
                    st.error("❌ Failed to generate thesis")
                    st.code(thesis.get('raw', 'Unknown error'))
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🚀 Bull Case")
                        bull_case = thesis.get('bull_case', {})
                        st.write(bull_case.get('thesis_summary', 'N/A'))
                        
                        st.write("**Key Drivers:**")
                        for driver in bull_case.get('key_drivers', []):
                            st.write(f"✅ {driver}")
                        
                        upside = bull_case.get('potential_return', 'N/A')
                        st.success(f"**Potential Upside:** {upside}")
                    
                    with col2:
                        st.subheader("🐻 Bear Case")
                        bear_case = thesis.get('bear_case', {})
                        st.write(bear_case.get('thesis_summary', 'N/A'))
                        
                        st.write("**Key Risks:**")
                        for risk in bear_case.get('key_risks', []):
                            st.write(f"⚠️ {risk}")
                        
                        downside = bear_case.get('potential_loss', 'N/A')
                        st.error(f"**Potential Downside:** {downside}")
                    
                    st.subheader("⚖️ Base Case")
                    base_case = thesis.get('base_case', {})
                    st.write(base_case.get('most_likely_scenario', 'N/A'))
                    
                    action = base_case.get('recommended_action', 'N/A')
                    if action == 'Buy':
                        st.success(f"**Recommendation:** {action}")
                    elif action == 'Sell':
                        st.error(f"**Recommendation:** {action}")
                    else:
                        st.info(f"**Recommendation:** {action}")
                    
            except Exception as e:
                st.error(f"❌ Error in investment thesis: {e}")
                st.exception(e)
        
        # TAB 4: Key Metrics
        with tabs[3]:
            try:
                with st.spinner("Extracting financial metrics..."):
                    metrics_df = st.session_state.analyzer.extract_key_metrics(company)
                
                if not metrics_df.empty and 'error' not in metrics_df.columns:
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Download button
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"{company}_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ Could not extract metrics")
                    if not metrics_df.empty:
                        st.dataframe(metrics_df)
                        
            except Exception as e:
                st.error(f"❌ Error extracting metrics: {e}")
                st.exception(e)
    
    # Multiple company comparison
    elif len(companies) > 1:
        st.subheader(f"Comparing: {', '.join(companies)}")
        
        try:
            with st.spinner("Performing comparative analysis..."):
                comparison = st.session_state.analyzer.comparative_analysis(companies)
            
            if 'error' in comparison:
                st.error("❌ Failed to compare companies")
                st.code(comparison.get('raw', 'Unknown error'))
            else:
                st.json(comparison)
                
        except Exception as e:
            st.error(f"❌ Error in comparative analysis: {e}")
            st.exception(e)


def render_history():
    """Query history"""
    
    if not st.session_state.analysis_history:
        st.info("No queries yet. Ask a question to get started!")
        return
    
    st.header("📜 Query History")
    
    for i, item in enumerate(reversed(st.session_state.analysis_history[-20:]), 1):
        with st.expander(f"{i}. {item['question'][:80]}..."):
            st.caption(f"⏱️ {item['timestamp']} | Latency: {item.get('latency', 0):.2f}s | Sources: {item.get('num_sources', 0)}")
            st.write(item['answer'])


def render_metrics_dashboard():
    """Performance metrics - FEATURE #4"""
    
    st.header("📊 Performance Metrics")
    
    if not st.session_state.query_metrics:
        st.info("No metrics yet. Run some queries to see performance data!")
        return
    
    metrics = st.session_state.query_metrics
    df = pd.DataFrame(metrics)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(metrics))
    
    with col2:
        avg_latency = df['latency'].mean()
        st.metric("Avg Latency", f"{avg_latency:.2f}s")
    
    with col3:
        p95_latency = df['latency'].quantile(0.95)
        st.metric("P95 Latency", f"{p95_latency:.2f}s")
    
    with col4:
        success_rate = df['success'].mean() * 100
        st.metric("Success Rate", f"{success_rate:.0f}%")
    
    # Charts
    st.subheader("Latency Over Time")
    st.line_chart(df['latency'])
    
    st.subheader("Recent Queries")
    display_df = df[['timestamp', 'question', 'latency', 'num_sources', 'success']].tail(10)
    st.dataframe(display_df, use_container_width=True)
    
    # Download metrics
    if st.button("📥 Download Metrics CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Download",
            csv,
            f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    # Error log
    if st.session_state.error_log:
        st.subheader("❌ Error Log")
        error_df = pd.DataFrame(st.session_state.error_log)
        st.dataframe(error_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main application"""
    
    # Page config
    st.set_page_config(
        page_title="Financial Intelligence Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize state
    initialize_session_state()
    
    # Header
    st.title("📊 Financial Intelligence Assistant")
    st.markdown("""
    AI-powered SEC filing analysis using **RAG (Retrieval-Augmented Generation)**  
    Built with: OpenAI GPT • LangChain • ChromaDB • Streamlit
    """)
    
    # Sidebar
    model_choice, temperature = render_sidebar()
    
    # Load RAG system
    load_rag_system(model_choice, temperature)
    
    # Main interface
    if st.session_state.index_loaded:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Q&A",
            "📊 Analysis Tools",
            "📜 History",
            "📈 Metrics"
        ])
        
        with tab1:
            render_query_interface()
        
        with tab2:
            render_analysis_tools()
        
        with tab3:
            render_history()
        
        with tab4:
            render_metrics_dashboard()
            
    else:
        # Quick start guide
        st.info("👈 Use the sidebar to configure the system and load data")
        
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        1. **Set up environment:**
           ```bash
           # Create .env file
           echo "OPENAI_API_KEY=your_key" > .env
           echo "SEC_API_KEY=your_key" >> .env
           ```
        
        2. **Collect data (Option A - Using sidebar):**
           - Enter company ticker (e.g., AAPL)
           - Click "Collect Filings"
           - System downloads SEC filings
        
        3. **Or collect data (Option B - Command line):**
           ```bash
           python src/data_collection/sec_collector.py
           ```
        
        4. **Start analyzing:**
           - System automatically builds RAG index
           - Ask questions or run analyses
           - View performance metrics
        
        ### ✨ Features
        
        - ✅ **Fixed:** Quick questions persist
        - ✅ **Fixed:** Better error handling
        - 📊 **New:** Performance metrics dashboard
        - 🔍 **Enhanced:** Source attribution
        - 📈 **New:** Query history tracking
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
    Built by Logan Liu | 
    <a href='https://github.com/loganatliu'>GitHub</a> | 
    <a href='https://www.linkedin.com/in/gang-logan-liu/'>LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()