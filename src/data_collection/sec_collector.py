"""
Financial Intelligence Assistant - Module 1: Data Collection (UPDATED)
File: src/data_collection/sec_collector.py

PURPOSE: Fetch and process SEC filings with CLEAN TEXT extraction
UPDATED: Now uses SEC-API ExtractorApi for clean, structured text

WHAT CHANGED:
- OLD: RenderApi → returned HTML/XML garbage
- NEW: ExtractorApi → returns CLEAN TEXT from specific sections

INSTALL: pip install sec-api
API KEY: Get from https://sec-api.io (100 free extractions, then ~$0.01-0.05 each)

INTERVIEW KEY: Shows you understand working with real financial data APIs
and making informed technical decisions when initial approach doesn't work.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from sec_api import QueryApi, ExtractorApi


class SECDataCollector:
    """
    Collects and processes SEC filings using sec-api.io ExtractorApi
    
    WHY EXTRACTORAPI?
    - Returns CLEAN TEXT (not HTML)
    - Can extract specific sections (Business, Risk Factors, MD&A)
    - Professional-grade text quality
    - Handles complex SEC filing formats
    
    INTERVIEW TIP: "I initially used RenderApi but it returned HTML.
    I switched to ExtractorApi which provides clean, section-specific
    text extraction. This demonstrates evaluating tools and adapting
    when the initial approach doesn't meet requirements."
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the SEC API clients
        
        TWO APIS USED:
        - QueryApi: Search and find filings
        - ExtractorApi: Extract clean text from specific sections
        """
        self.query_api = QueryApi(api_key=api_key)
        self.extractor_api = ExtractorApi(api_key=api_key)
        self.data_dir = "data\\raw"
        os.makedirs(self.data_dir, exist_ok=True)
        print("✅ SEC Data Collector initialized (ExtractorApi)")
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Make filename Windows-compatible by removing invalid characters
        
        WINDOWS FILENAME RESTRICTIONS:
        - Cannot contain: < > : " / \ | ? *
        - Colons are particularly problematic in timestamps
        
        SOLUTION: Replace invalid characters with underscores
        """
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename
    
    def fetch_company_filings(
        self, 
        ticker: str, 
        filing_types: List[str] = ["10-K", "10-Q"],
        limit: int = 10,
        years: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch recent filings for a company
        
        PARAMETERS EXPLAINED:
        - ticker: Stock symbol (e.g., "AAPL", "MSFT")
        - filing_types: ["10-K"] = Annual reports, ["10-Q"] = Quarterly reports
        - limit: Number of recent filings to retrieve
        - years: If specified, fetch filings from last N years
        
        INTERVIEW POINT: Why these filing types?
        - 10-K: Comprehensive annual business overview (most detailed)
        - 10-Q: Quarterly updates with recent performance
        - 8-K: Material events (acquisitions, leadership changes)
        
        DATA COVERAGE:
        - 3 filings (10-K only) = 3 years of annual data
        - 5 filings (10-K only) = 5 years of annual data
        - 20 filings (10-Q) = 5 years of quarterly data
        
        RETURNS: List of filing metadata (URL, date, type, etc.)
        """
        
        # Build the SEC EDGAR query
        query = f"ticker:{ticker} AND formType:({' OR '.join(filing_types)})"
        
        # Add date filter if years specified
        if years:
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=365*years)).strftime("%Y-%m-%d")
            query += f' AND filedAt:[{start_date} TO *]'
            print(f"🔍 Searching SEC EDGAR for {ticker} filings from last {years} years...")
        else:
            print(f"🔍 Searching SEC EDGAR for {ticker} filings...")
        
        try:
            # Execute the search query
            filings = self.query_api.get_filings({
                "query": query,
                "from": "0",
                "size": str(limit),
                "sort": [{"filedAt": {"order": "desc"}}]
            })
            
            results = filings.get("filings", [])
            print(f"✅ Found {len(results)} filings for {ticker}")
            
            # Show date range
            if results:
                oldest = results[-1]['filedAt'][:10]
                newest = results[0]['filedAt'][:10]
                print(f"   Date range: {oldest} to {newest}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error fetching filings: {e}")
            return []
    
    def extract_10k_sections(self, filing_url: str, ticker: str) -> str:
        """
        Extract key sections from a 10-K filing as CLEAN TEXT
        
        SECTIONS EXTRACTED:
        - Item 1: Business Description
        - Item 1A: Risk Factors  
        - Item 7: Management's Discussion & Analysis (MD&A)
        - Item 8: Financial Statements
        
        WHY THESE SECTIONS?
        - Business: Company overview, products, strategy
        - Risk Factors: Comprehensive risk assessment
        - MD&A: Management's analysis of performance
        - Financials: Income statement, balance sheet
        
        RETURNS: Combined text from all sections
        """
        
        print(f"   📖 Extracting clean text from sections...")
        
        # Section codes for 10-K filings
        sections = {
            "1": "BUSINESS DESCRIPTION",
            "1A": "RISK FACTORS",
            "7": "MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "8": "FINANCIAL STATEMENTS"
        }
        
        extracted_text = f"{ticker.upper()} - Form 10-K\n"
        extracted_text += f"Source: {filing_url}\n"
        extracted_text += f"{'='*60}\n\n"
        
        for section_code, section_name in sections.items():
            try:
                print(f"      • Extracting {section_name}...")
                
                # Extract section as TEXT (not HTML!)
                section_text = self.extractor_api.get_section(
                    filing_url, 
                    section_code, 
                    "text"  # This is KEY - returns text, not HTML
                )
                
                if section_text and len(section_text) > 100:
                    extracted_text += f"\n{'='*60}\n"
                    extracted_text += f"ITEM {section_code}: {section_name}\n"
                    extracted_text += f"{'='*60}\n\n"
                    extracted_text += section_text
                    extracted_text += "\n\n"
                    print(f"      ✅ {section_name}: {len(section_text):,} chars")
                else:
                    print(f"      ⚠️  {section_name}: No content or too short")
                    
            except Exception as e:
                print(f"      ⚠️  Could not extract {section_name}: {e}")
        
        return extracted_text
    
    def extract_10q_sections(self, filing_url: str, ticker: str) -> str:
        """
        Extract key sections from a 10-Q filing as CLEAN TEXT
        
        SECTIONS FOR 10-Q (different from 10-K):
        - Part I, Item 1: Financial Statements
        - Part I, Item 2: MD&A
        - Part II, Item 1A: Risk Factors (if updated)
        
        10-Q sections use different codes than 10-K
        """
        
        print(f"   📖 Extracting clean text from 10-Q sections...")
        
        # Section codes for 10-Q filings (different structure than 10-K)
        sections = {
            "part1item1": "FINANCIAL STATEMENTS",
            "part1item2": "MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "part2item1a": "RISK FACTORS"
        }
        
        extracted_text = f"{ticker.upper()} - Form 10-Q\n"
        extracted_text += f"Source: {filing_url}\n"
        extracted_text += f"{'='*60}\n\n"
        
        for section_code, section_name in sections.items():
            try:
                print(f"      • Extracting {section_name}...")
                
                # Extract section as TEXT
                section_text = self.extractor_api.get_section(
                    filing_url, 
                    section_code, 
                    "text"
                )
                
                if section_text and len(section_text) > 100:
                    extracted_text += f"\n{'='*60}\n"
                    extracted_text += f"{section_name}\n"
                    extracted_text += f"{'='*60}\n\n"
                    extracted_text += section_text
                    extracted_text += "\n\n"
                    print(f"      ✅ {section_name}: {len(section_text):,} chars")
                else:
                    print(f"      ⚠️  {section_name}: No content or too short")
                    
            except Exception as e:
                print(f"      ⚠️  Could not extract {section_name}: {e}")
        
        return extracted_text
    
    def download_filing_text(
        self, 
        filing_url: str, 
        filing_id: str, 
        filing_type: str,
        ticker: str
    ) -> Optional[str]:
        """
        Download and extract CLEAN TEXT from a filing
        
        KEY CHANGE FROM OLD VERSION:
        - OLD: RenderApi → HTML output
        - NEW: ExtractorApi → Clean text from specific sections
        
        ADVANTAGES:
        - No HTML tags or CSS
        - Structured by section
        - Better quality for RAG/LLM processing
        """
        
        try:
            print(f"📥 Downloading filing {filing_id}...")
            
            # Extract clean text based on filing type
            if filing_type == "10-K":
                filing_text = self.extract_10k_sections(filing_url, ticker)
            elif filing_type == "10-Q":
                filing_text = self.extract_10q_sections(filing_url, ticker)
            else:
                print(f"   ⚠️  Unsupported filing type: {filing_type}")
                return None
            
            # Check if we got meaningful content
            if len(filing_text) < 1000:
                print(f"   ⚠️  Filing text too short ({len(filing_text)} chars), skipping...")
                return None
            
            # Save to disk for reuse
            filepath = os.path.join(self.data_dir, f"{filing_id}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(filing_text)
            
            print(f"   ✅ Saved to {filepath} ({len(filing_text):,} characters)\n")
            return filing_text
            
        except Exception as e:
            print(f"   ❌ Error downloading filing: {e}\n")
            return None
    
    def process_company(
        self, 
        ticker: str, 
        num_filings: int = 3,
        years: Optional[int] = None,
        filing_types: List[str] = ["10-K", "10-Q"]
    ) -> Dict:
        """
        Complete pipeline: Fetch and download all filings for a company
        
        PARAMETERS:
        - ticker: Stock symbol
        - num_filings: Max number of filings to download
        - years: If specified, fetch filings from last N years
        - filing_types: Types of filings to fetch
        
        USAGE EXAMPLES:
        
        # Get last 3 filings (default - covers ~1-2 years)
        collector.process_company("AAPL", num_filings=3)
        
        # Get 5 years of annual reports
        collector.process_company("AAPL", num_filings=5, filing_types=["10-K"])
        
        # Get 5 years of quarterly data (20 quarters)
        collector.process_company("AAPL", num_filings=20, filing_types=["10-Q"])
        
        # Get all filings from last 5 years
        collector.process_company("AAPL", years=5, num_filings=50)
        
        PRODUCTION PATTERN:
        1. Fetch metadata (cheap API call)
        2. Download only what's needed (ExtractorApi call - uses quota)
        3. Store locally to avoid redundant downloads
        4. Return structured data for next pipeline stage
        
        COST CONSIDERATION:
        - QueryApi: Free/cheap
        - ExtractorApi: ~$0.01-0.05 per extraction
        - For 5 years (5 annual reports) = 5 extractions = ~$0.15-0.25
        
        INTERVIEW QUESTION: "How do you decide how much historical data to collect?"
        ANSWER: 
        - Balance between coverage and cost
        - For most financial analysis: 3-5 years is sufficient
        - For trend analysis: 5-10 years
        - For full historical: Would use different approach (database)
        """
        
        print(f"\n{'='*60}")
        print(f"📊 Processing {ticker}")
        print(f"{'='*60}\n")
        
        # Step 1: Fetch filing list
        filings = self.fetch_company_filings(
            ticker, 
            filing_types=filing_types,
            limit=num_filings,
            years=years
        )
        
        if not filings:
            return {"ticker": ticker, "filings": [], "error": "No filings found"}
        
        # Step 2: Download each filing with clean text extraction
        processed_filings = []
        for filing in filings:
            # Create filing ID and sanitize for Windows
            filing_id = f"{ticker}_{filing['formType']}_{filing['filedAt']}"
            filing_id_safe = self.sanitize_filename(filing_id)
            
            filing_url = filing['linkToFilingDetails']
            filing_type = filing['formType']
            
            # Check if already downloaded (save API calls and money!)
            filepath = os.path.join(self.data_dir, f"{filing_id_safe}.txt")
            if os.path.exists(filepath):
                print(f"⏭️  Already have {filing_id_safe}, skipping...\n")
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = self.download_filing_text(
                    filing_url, 
                    filing_id_safe, 
                    filing_type,
                    ticker
                )
                time.sleep(1)  # Rate limiting (be respectful)
            
            if text:
                processed_filings.append({
                    "id": filing_id_safe,
                    "type": filing['formType'],
                    "date": filing['filedAt'],
                    "ticker": ticker,
                    "filepath": filepath,
                    "text_length": len(text),
                    "text_preview": text[:500]
                })
        
        return {
            "ticker": ticker,
            "filings": processed_filings,
            "total_downloaded": len(processed_filings)
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example usage of the SEC data collector with ExtractorApi
    
    TO RUN THIS:
    1. Get API key from https://sec-api.io
    2. Add to .env file: SEC_API_KEY=your_key_here
    3. Run: python src\\data_collection\\sec_collector.py
    
    DATA COLLECTION STRATEGIES:
    
    Strategy 1: Recent snapshot (RECOMMENDED FOR PORTFOLIO)
    - 2 companies × 3 filings each = 6 extractions
    - Cost: ~$0.18 (within free tier)
    - Coverage: Last 1-2 years
    - Good for: Demo, testing, portfolio projects
    
    Strategy 2: 5-year historical (FOR COMPREHENSIVE ANALYSIS)
    - 2 companies × 5 annual reports = 10 extractions
    - Cost: ~$0.30-0.50
    - Coverage: 5 years of annual data
    - Good for: Trend analysis, historical comparisons
    
    Strategy 3: Full quarterly data (FOR DETAILED TIME-SERIES)
    - 2 companies × 20 quarterly reports = 40 extractions
    - Cost: ~$1.20-2.00
    - Coverage: 5 years of quarterly data
    - Good for: Detailed performance tracking
    """
    
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n" + "="*60)
    print("🚀 SEC EDGAR DATA COLLECTOR")
    print("="*60)
    print("Using SEC-API ExtractorApi (CLEAN TEXT OUTPUT)")
    print("="*60 + "\n")
    
    # Initialize collector
    api_key = os.getenv("SEC_API_KEY")
    if not api_key:
        print("❌ Error: SEC_API_KEY not found in .env file")
        print("   Get your API key at: https://sec-api.io")
        print("   Add to .env: SEC_API_KEY=your_key_here")
        return
    
    collector = SECDataCollector(api_key)
    
    # ================================================================
    # CHOOSE YOUR COLLECTION STRATEGY
    # ================================================================
    
    print("📋 Collection Strategy Options:")
    print("1. Recent snapshot (3 filings, ~$0.18)")
    print("2. 5-year annual data (5 filings, ~$0.30)")
    print("3. 5-year quarterly data (20 filings, ~$1.50)")
    print()
    
    # For portfolio/demo: Use Strategy 1 (Recent)
    # For comprehensive analysis: Use Strategy 2 (5-year annual)
    
    # STRATEGY 1: Recent snapshot (DEFAULT - RECOMMENDED)
    companies = ["AAPL", "MSFT"]
    results = []
    
    for ticker in companies:
        result = collector.process_company(
            ticker, 
            num_filings=3,  # Last 3 filings
            filing_types=["10-K", "10-Q"]  # Mix of annual and quarterly
        )
        results.append(result)
        print(f"✅ Processed {result['total_downloaded']} filings for {ticker}\n")
    
    # STRATEGY 2: 5-year annual data (UNCOMMENT TO USE)
    # companies = ["AAPL", "MSFT"]
    # results = []
    # 
    # for ticker in companies:
    #     result = collector.process_company(
    #         ticker, 
    #         num_filings=5,  # Last 5 annual reports
    #         filing_types=["10-K"]  # Annual reports only
    #     )
    #     results.append(result)
    #     print(f"✅ Processed {result['total_downloaded']} filings for {ticker}\n")
    
    # STRATEGY 3: 5-year quarterly data (UNCOMMENT TO USE)
    # companies = ["AAPL", "MSFT"]
    # results = []
    # 
    # for ticker in companies:
    #     result = collector.process_company(
    #         ticker, 
    #         num_filings=20,  # Last 20 quarters = 5 years
    #         filing_types=["10-Q"]  # Quarterly reports only
    #     )
    #     results.append(result)
    #     print(f"✅ Processed {result['total_downloaded']} filings for {ticker}\n")
    
    # Save summary
    summary_file = "data\\processed\\collection_summary.json"
    os.makedirs("data\\processed", exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print("="*60)
    print("✅ COLLECTION COMPLETE!")
    print("="*60)
    
    total_files = sum(r['total_downloaded'] for r in results)
    print(f"\nTotal files collected: {total_files}")
    
    print("\nFiles created:")
    for r in results:
        for f in r.get('filings', []):
            print(f"  • {f['id']}.txt ({f['text_length']:,} chars)")
    
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Verify clean text: ")
    print("   powershell -command \"Get-Content 'data\\raw\\AAPL_10-K*.txt' | Select-Object -First 50\"")
    print("\n2. Delete old index: rmdir /s /q data\\chroma_db")
    print("\n3. Rebuild RAG index: python src\\rag_engine\\financial_rag.py")
    print("\n4. Run Streamlit app: streamlit run app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


"""
=============================================================================
INTERVIEW PREPARATION GUIDE - UPDATED
=============================================================================

KEY STORY TO TELL:

"I initially implemented data collection using sec-api's RenderApi, but 
discovered it returned raw HTML/XML, requiring complex parsing. I evaluated 
alternatives and switched to the ExtractorApi, which provides clean, 
section-specific text extraction. This increased API costs slightly (~$0.03 
per filing vs free), but dramatically improved data quality and reduced 
preprocessing complexity. This demonstrates pragmatic engineering - investing 
in data quality upfront to simplify downstream processing."

TECHNICAL QUESTIONS:

1. "Why did you switch from RenderApi to ExtractorApi?"
   → RenderApi: Raw HTML with CSS/styling → requires BeautifulSoup parsing
   → ExtractorApi: Clean text by section → ready for RAG/LLM processing
   → Trade-off: Slight cost increase for better quality and less code

2. "How do you handle the cost of ExtractorApi?"
   → Cache extracted text locally (check before downloading)
   → Extract only necessary sections (Business, Risk Factors, MD&A)
   → Use for production, mock data for testing
   → ~$0.03 per extraction is negligible vs developer time saved

3. "What sections do you extract and why?"
   → 10-K Item 1: Business description and strategy
   → 10-K Item 1A: Comprehensive risk factors
   → 10-K Item 7: MD&A - management's analysis of performance
   → 10-K Item 8: Financial statements and data
   → These contain 80% of useful information for financial analysis

4. "How would you optimize this for 1000 companies?"
   → Parallel processing (async downloads)
   → Incremental updates (only new filings)
   → Database storage (PostgreSQL) vs flat files
   → Separate extraction from storage layers
   → Consider cheaper alternatives for bulk historical data

5. "What about data quality validation?"
   → Check extracted text length (min 1000 chars)
   → Verify section extraction success rate
   → Sample review of extracted content
   → Monitor API error rates
   → Log failed extractions for review

COST BREAKDOWN:

Development/Testing:
- 3 companies × 3 filings = 9 extractions
- 9 × $0.03 = $0.27 (well within free tier of 100 extractions)

Production (monthly):
- 50 companies × 4 quarterly updates = 200 extractions
- 200 × $0.03 = $6/month
- Minimal cost for high-quality data

ALTERNATIVES CONSIDERED:

1. EdgarTools (FREE):
   + No cost, good text quality
   - Slightly more code to manage
   - Rate limited by SEC directly

2. SEC-API ExtractorApi (PAID):
   + Best text quality
   + Section-specific extraction
   + Professional-grade
   - Small cost (~$0.03/extraction)

3. Web Scraping:
   + Free
   - Legal/ethical concerns
   - Fragile (breaks when HTML changes)
   - Rate limiting issues

DECISION: ExtractorApi for portfolio project - shows understanding of 
data quality trade-offs and willingness to invest in better tools.

=============================================================================
"""