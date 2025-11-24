"""
Financial Intelligence Assistant - Module 2: RAG Engine
File: src/rag_engine/financial_rag.py

PURPOSE: Build a Retrieval-Augmented Generation system for financial analysis
THIS IS THE CORE OF YOUR AI SYSTEM - Master this and you'll ace interviews!

CONCEPTS YOU'LL LEARN:
- Vector embeddings and semantic search
- RAG architecture (retrieve → augment → generate)
- LangChain framework patterns
- Production LLM system design

INTERVIEW GOLD: This module demonstrates you understand modern AI architecture

UPDATED: Compatible with LangChain 0.2+ (latest version)
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# LangChain imports - updated for version 0.2+
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class FinancialRAGEngine:
    """
    RAG (Retrieval-Augmented Generation) system for financial documents
    
    ═══════════════════════════════════════════════════════════════════
    WHAT IS RAG? (Critical interview concept!)
    ═══════════════════════════════════════════════════════════════════
    
    Traditional LLM: Question → LLM → Answer
    Problem: Limited by training data, hallucinations
    
    RAG System: Question → Retrieve Relevant Docs → LLM + Docs → Answer
    Benefits: 
    - Grounded in actual documents (reduces hallucinations)
    - Up-to-date information (not limited to training cutoff)
    - Traceable answers (can cite sources)
    
    ═══════════════════════════════════════════════════════════════════
    RAG PIPELINE STAGES:
    ═══════════════════════════════════════════════════════════════════
    
    1. CHUNKING: Split documents into manageable pieces
       Why? LLMs have token limits (e.g., 8K, 32K tokens)
       
    2. EMBEDDING: Convert text to vectors (arrays of numbers)
       Why? Enables semantic similarity search
       
    3. STORAGE: Store vectors in database for fast retrieval
       Why? Need efficient similarity search across millions of docs
       
    4. RETRIEVAL: Find most relevant chunks for a query
       How? Cosine similarity between query vector and doc vectors
       
    5. GENERATION: LLM generates answer using retrieved context
       How? Prompt engineering with retrieved docs as context
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.0
    ):
        """
        Initialize the RAG engine
        
        PARAMETER DEEP DIVE:
        
        model_name: Which LLM to use for generation
        - "gpt-4": Most capable, expensive (~$0.03/1K tokens)
        - "gpt-3.5-turbo": Fast and cheap (~$0.001/1K tokens)
        - For production: Start cheap, upgrade where needed
        
        embedding_model: Which model to convert text → vectors
        - "text-embedding-3-small": 1536 dimensions, fast
        - "text-embedding-3-large": 3072 dimensions, more accurate
        - Trade-off: Accuracy vs speed vs cost
        
        chunk_size: Characters per document chunk
        - 1000: ~250 tokens (good for detailed info)
        - Too small: Loses context
        - Too large: Retrieval is less precise
        
        chunk_overlap: Characters shared between chunks
        - 200: Ensures sentences aren't cut mid-context
        - Prevents information loss at boundaries
        
        temperature: LLM creativity (0.0 to 1.0)
        - 0.0: Deterministic, factual (best for financial analysis)
        - 1.0: Creative, varied (better for brainstorming)
        
        INTERVIEW TIP: Be ready to explain these trade-offs!
        """
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        # RecursiveCharacterTextSplitter: Smart splitting at sentence boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # How to measure chunk size
            separators=["\n\n", "\n", ". ", " ", ""]  # Try these in order
        )
        
        # Initialize embeddings model
        # This converts text to 1536-dimensional vectors
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize LLM for generation
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature  # 0.0 for factual, deterministic outputs
        )
        
        # Vector store (starts empty, populated in build_index())
        self.vectorstore = None
        self.retriever = None
    
    def load_documents(self, filepath: str) -> List[Document]:
        """
        Load and chunk a financial document
        
        DOCUMENT OBJECT: LangChain's standard data structure
        - page_content: The actual text
        - metadata: Info about the document (source, date, etc.)
        
        WHY METADATA MATTERS:
        - Enables filtering (e.g., "only 2023 filings")
        - Provides source attribution in answers
        - Helps with debugging and monitoring
        """
        
        print(f"📄 Loading document: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create LangChain Document object
        doc = Document(
            page_content=text,
            metadata={
                "source": filepath,
                "filename": os.path.basename(filepath),
                "load_date": str(os.path.getmtime(filepath))
            }
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        print(f"✂️  Split into {len(chunks)} chunks")
        print(f"📊 Chunk size: {self.chunk_size} chars, overlap: {self.chunk_overlap}")
        
        return chunks
    
    def build_index(self, document_paths: List[str], persist_directory: str = "data\\chroma_db"):
        """
        Build the vector index from documents
        
        ═══════════════════════════════════════════════════════════════
        THIS IS THE HEART OF RAG! Understand this deeply.
        ═══════════════════════════════════════════════════════════════
        
        STEP-BY-STEP PROCESS:
        
        1. Load all documents and chunk them
        2. Generate embeddings for each chunk
           - Embedding: text → [0.23, -0.45, 0.67, ...] (1536 numbers)
           - Similar text has similar vectors (cosine similarity)
        3. Store in ChromaDB (vector database)
           - Optimized for similarity search
           - Uses HNSW algorithm (approximate nearest neighbors)
        4. Create retriever for querying
        
        INTERVIEW QUESTIONS TO PREPARE:
        
        Q: "Why do we need a vector database? Why not just search text?"
        A: Vector search finds semantic similarity, not just keywords.
           Example: "revenue growth" matches "increasing sales" 
        
        Q: "How does similarity search work?"
        A: Cosine similarity between vectors. Closer vectors = more similar.
           cos(A, B) = (A · B) / (||A|| * ||B||)
        
        Q: "What's the time complexity of similarity search?"
        A: Exact: O(n), Approximate (HNSW): O(log n)
           For million+ docs, approximate is essential
        """
        
        print(f"\n{'='*60}")
        print("🏗️  BUILDING RAG INDEX")
        print(f"{'='*60}\n")
        
        # Step 1: Load and chunk all documents
        all_chunks = []
        for path in document_paths:
            chunks = self.load_documents(path)
            all_chunks.extend(chunks)
        
        print(f"\n📚 Total chunks to index: {len(all_chunks)}")
        print(f"💰 Estimated embedding cost: ${len(all_chunks) * 0.00002:.4f}")
        
        # Step 2 & 3: Generate embeddings and store in ChromaDB
        print("🔄 Generating embeddings and building index...")
        
        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory  # Save to disk for reuse
        )
        
        # Step 4: Create retriever
        # k=4 means "return top 4 most relevant chunks"
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Number of chunks to retrieve
        )
        
        print(f"✅ Index built! Stored in {persist_directory}")
        print(f"🔍 Retriever ready: top-4 similarity search enabled\n")
    
    def load_existing_index(self, persist_directory: str = "data\\chroma_db"):
        """
        Load a previously built index (avoid rebuilding)
        
        PRODUCTION PATTERN:
        - Build index once (expensive)
        - Load index many times (cheap)
        - Rebuild only when documents change
        """
        
        print(f"📂 Loading existing index from {persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        print("✅ Index loaded successfully!\n")
    
    def format_docs(self, docs):
        """Helper function to format retrieved documents"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_rag_chain(self):
        """
        Create a RAG chain using LangChain Expression Language (LCEL)
        
        ═══════════════════════════════════════════════════════════════
        LCEL: Modern LangChain pattern (v0.2+)
        ═══════════════════════════════════════════════════════════════
        
        Instead of the old RetrievalQA chain, we use LCEL which is:
        - More flexible
        - Better performance
        - Easier to customize
        - Standard in LangChain 0.2+
        
        Pattern: retriever | format | prompt | llm | parse
        """
        
        # Custom prompt template for financial analysis
        template = """You are an expert financial analyst with deep knowledge of SEC filings, 
financial statements, and investment analysis. 

Use the following pieces of context from financial documents to answer the question at the end. 
If you don't know the answer based on the provided context, say so - don't make up information.

When discussing financial metrics:
- Provide specific numbers and percentages when available
- Compare to prior periods when relevant
- Identify trends and patterns
- Cite which document/section the information comes from

Context from financial documents:
{context}

Question: {question}

Detailed Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Create RAG chain using LCEL
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG system
        
        EXECUTION FLOW:
        1. Convert question to embedding vector
        2. Find top-k most similar document chunks (retrieval)
        3. Inject chunks into prompt as context (augmentation)
        4. LLM generates answer (generation)
        5. Return answer + source documents
        
        INTERVIEW QUESTION: "How do you handle queries that span multiple docs?"
        ANSWER: The retriever can pull chunks from different documents.
                The LLM synthesizes information across all retrieved chunks.
        """
        
        if not self.retriever:
            return {"error": "Index not built. Call build_index() first."}
        
        print(f"\n🤔 Question: {question}")
        print("🔍 Retrieving relevant documents...")
        
        # Create RAG chain
        rag_chain = self.create_rag_chain()
        
        # Get relevant documents for source attribution
        relevant_docs = self.retriever.invoke(question)
        
        # Execute query
        answer = rag_chain.invoke(question)
        
        response = {
            "question": question,
            "answer": answer,
            "sources": []
        }
        
        # Add source information
        if return_sources:
            for i, doc in enumerate(relevant_docs, 1):
                response["sources"].append({
                    "source_num": i,
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "excerpt": doc.page_content[:200] + "..."
                })
        
        return response


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example: Build RAG system and query financial documents
    
    WORKFLOW:
    1. Build index from downloaded SEC filings
    2. Query for financial insights
    3. Get answers with source citations
    """
    
    print(f"\n{'='*60}")
    print("🚀 FINANCIAL RAG ENGINE DEMO")
    print(f"{'='*60}\n")
    
    # Initialize RAG engine
    rag = FinancialRAGEngine(
        model_name="gpt-3.5-turbo",  # Works for all accounts, fast & cheap
        chunk_size=1000,
        chunk_overlap=200,
        temperature=0.0  # Factual answers only
    )
    
    # Get all downloaded filings
    data_dir = "data\\raw"
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        print("   Creating data directory...")
        os.makedirs(data_dir, exist_ok=True)
        print("   Run Module 1 (sec_collector.py) first to download data!")
        return
        
    document_paths = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.txt')
    ]
    
    if not document_paths:
        print("❌ No documents found in data\\raw\\")
        print("   Run Module 1 (sec_collector.py) first!")
        return
    
    # Build or load index
    persist_dir = "data\\chroma_db"
    if os.path.exists(persist_dir):
        print("📂 Found existing index, attempting to load...")
        try:
            rag.load_existing_index(persist_dir)
            # Test if index has documents
            test_docs = rag.vectorstore._collection.count()
            if test_docs == 0:
                print("⚠️  Index is empty, rebuilding...")
                rag.build_index(document_paths, persist_dir)
            else:
                print(f"✅ Index has {test_docs} documents")
        except Exception as e:
            print(f"⚠️  Error loading index: {e}")
            print("🔄 Rebuilding index...")
            rag.build_index(document_paths, persist_dir)
    else:
        print("🔨 No index found, building new one...")
        rag.build_index(document_paths, persist_dir)
    
    # Example queries
    questions = [
        "What was the revenue growth rate in the most recent quarter?",
        "What are the main risk factors mentioned in the filings?",
        "How has R&D spending changed over the past year?",
        "What are the company's strategic priorities?"
    ]
    
    for question in questions:
        result = rag.query(question)
        
        print(f"\n{'─'*60}")
        print(f"❓ Q: {result['question']}")
        print(f"\n💡 A: {result['answer']}")
        print(f"\n📚 Sources:")
        for src in result['sources']:
            print(f"   {src['source_num']}. {src['filename']}")
        print(f"{'─'*60}")


if __name__ == "__main__":
    main()


"""
=============================================================================
INTERVIEW PREPARATION - RAG SYSTEMS
=============================================================================

🔥 CRITICAL CONCEPTS TO MASTER:

1. WHAT IS RAG?
   "RAG combines retrieval with generation. Instead of relying solely on 
   the LLM's training data, we retrieve relevant documents and use them 
   as context for generating answers."

2. WHY RAG OVER FINE-TUNING?
   - Fine-tuning: Expensive, static, requires retraining for updates
   - RAG: Cheap, dynamic, just add new documents to index
   - RAG reduces hallucinations by grounding in facts

3. VECTOR EMBEDDINGS EXPLAINED:
   "Embeddings convert text to numbers. Similar text gets similar vectors.
   We use cosine similarity to find relevant documents."
   
4. CHUNKING STRATEGY:
   "Too small: Loses context. Too large: Retrieval is imprecise.
   1000 chars with 200 overlap is a good starting point."

5. EVALUATION METRICS:
   - Retrieval: Precision@k, Recall@k, MRR
   - Generation: BLEU, ROUGE, Human evaluation
   - End-to-end: Answer accuracy, source citation accuracy

COMMON INTERVIEW QUESTIONS:

Q: "How do you handle multi-hop questions?" (e.g., "Compare revenue growth 
    between Company A and Company B")
A: Advanced: Use multiple retrieval rounds or chain-of-thought prompting
   Basic: Retrieve broader context, let LLM synthesize

Q: "What if the document is too long for the LLM context window?"
A: - Chunk more aggressively
   - Use map-reduce pattern (summarize chunks, then combine)
   - Consider hierarchical retrieval

Q: "How do you update the index when new docs arrive?"
A: - Incremental indexing (add new docs without rebuilding)
   - Scheduled batch updates
   - Real-time indexing for time-sensitive data

NEXT MODULE:
Build financial analysis tools that use the RAG engine to generate
investment insights, risk assessments, and comparative analysis.
=============================================================================
"""