# **Agentic Medical AI Assistant (Multi-Step RAG + LangGraph)**

An agentic AI system that performs multi-step medical reasoning using RAG (Retrieval-Augmented Generation), query decomposition, and adaptive search.

Built using LangGraph, local LLMs (Ollama), and vector search, this system simulates how a doctor gathers, validates, and synthesizes information before giving an answer.


## **Key Idea** 
Unlike traditional RAG systems (single query → single answer), this project:
1. Breaks a medical query into multiple sub-queries

2. Performs iterative retrieval and validation

3. Synthesizes a final grounded response

## **Why This Project?**

Example: "I have abdominal pain and vomiting"
A simple RAG system:-  Retrieves shallow information.

This system:
            Decomposes → researches → validates → synthesizes

## **System Architecture (Agentic Workflow)**
## Multi-Step Reasoning Pipeline
1. Query Decomposition Agent
  - Breaks user query into multiple research queries
2. Research Subgraph (Core Agent Loop)
  - Retrieve documents (ChromaDB)
  - Evaluate relevance (LLM-based filtering)
  - Fallback to web search if needed
3. Summarization Agent
  - Generates structured summaries per query
4. Final Synthesis
  - Combines all summaries into a coherent medical response

## **Tech Stack**
- LangGraph – Agent orchestration
- Ollama (phi3 / DeepSeek) – Local LLM inference
- ChromaDB – Vector database
- Streamlit – UI
- Python – Backend

## **How to Run**
1. Clone repo
    git clone https://github.com/Deepanjali-Sharma/medical-rag-agent
    cd agentic-medical-assistant

2. Setup environment
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

3. Add environment variables
  Create .env:
      TAVILY_API_KEY=your_key_here

4. Run Ollama
    ollama run phi3

5. Run app
    streamlit run app.py