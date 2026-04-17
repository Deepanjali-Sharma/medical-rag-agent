import datetime
from typing_extensions import Literal
from langgraph.graph import END
import json
import re
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from src.assistant.configuration import Configuration
from src.assistant.vector_db import get_or_create_vector_db
from src.assistant.state import MedicalState, MedicalStateInput, MedicalStateOutput, QuerySearchState, QuerySearchStateInput, QuerySearchStateOutput
from src.assistant.prompts import RESEARCH_QUERY_WRITER_PROMPT, RELEVANCE_EVALUATOR_PROMPT, SUMMARIZER_PROMPT, REPORT_WRITER_PROMPT
from src.assistant.utils import format_documents_with_metadata, invoke_llm, invoke_ollama, parse_output, tavily_search, Evaluation, Queries

# Number of query to process in parallel for each batch
BATCH_SIZE = 3

def generate_medical_queries(state: MedicalState, config: RunnableConfig):
    print("--- Generating medical queries ---")
    user_instructions = state["user_instructions"]
    max_queries = config["configurable"].get("max_search_queries", 3)
    
    query_writer_prompt = RESEARCH_QUERY_WRITER_PROMPT.format(
        max_queries=max_queries,
        date=datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
    )
    print("--> Calling LLM...")
    
    result = invoke_ollama(
        model='phi3',
        system_prompt=query_writer_prompt,
        user_prompt=f"""
        Patient symptoms / query: {user_instructions}

        Generate clinical search queries including:
        - possible diseases or conditions
        - recommended diagnostic tests
        - possible treatments

        Make the queries medically relevant and precise.
        """,
        output_format=Queries
    )

    print(" LLM returned:", result)
    return {"research_queries": result.queries}

def safe_parse(response):
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass


    return {}

def medical_reasoning(state: MedicalState):
    print("--- Medical reasoning ---")

    user_input = state["user_instructions"]

    reasoning_prompt = f"""
    You are a medical assistant.

    Extract symptoms and provide possible conditions.

    User input: {state["user_instructions"]}

    Return ONLY valid JSON. Do NOT add any explanation.

    Format:
    {{
        "symptoms": ["..."],
        "possible_conditions": ["..."],
        "severity_level": "low" | "medium" | "high",
        "risk_flags": ["..."]
    }}
    """

    result = invoke_ollama(
        model='phi3',
        system_prompt="You are a clinical reasoning assistant.",
        user_prompt=reasoning_prompt
    )

    parsed = safe_parse(result)

    return {
        "symptoms": parsed.get("symptoms", []),
        "possible_conditions": parsed.get("possible_conditions", []),
        "severity_level": parsed.get("severity_level", "low"),
        "risk_flags": parsed.get("risk_flags", [])
    }

def search_queries(state: MedicalState):
    print("--- Searching queries ---")

    current_position = state.get("current_position", 0)
    queries = state["research_queries"]

    if current_position >= len(queries):
        return {"done": True}

    query = queries[current_position]
    print("Current query:", query)
    
    return {
        **state,
        "current_position": current_position + 1,
        "query": str(query)
    }

def check_more_queries(state: MedicalState) -> Literal["search_queries", "generate_final_answer"]:
    """Check if there are more queries to process"""
    current_position = state.get("current_position", 0)
    if current_position < len(state["research_queries"]):
        return "search_queries"
    return "generate_final_answer"

# def initiate_query_research(state: MedicalState):
#     queries = state["research_queries"]
#     current_position = state["current_position"]

#     batch_end = min(current_position, len(queries))
#     current_batch = queries[current_position - BATCH_SIZE:batch_end]

#     print("DEBUG current_batch:", current_batch)

#     return [
#         Send("search_and_summarize_query", {"query": q})
#         for q in current_batch
#     ]

def retrieve_rag_documents(state: QuerySearchState):
    """Retrieve documents from the RAG database."""
    print("--- Retrieving documents ---")
    query = state["query"]
    vectorstore = get_or_create_vector_db()
    vectorstore_retreiver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    documents = vectorstore_retreiver.invoke(query)

    print(" Retrieved docs:", documents)
    return {
        "retrieved_docs": [doc.page_content for doc in documents] if documents else []
    }

def evaluate_retrieved_documents(state: QuerySearchState):
    query = state["query"]
    retrieved_documents = state.get("retrieved_docs", [])

    if not retrieved_documents:
        return {"are_documents_relevant": False}

    evaluation_prompt = RELEVANCE_EVALUATOR_PROMPT.format(
        query=query,
        documents="\n\n".join(retrieved_documents)
    )

    evaluation = invoke_ollama(
        model='phi3',
        system_prompt=evaluation_prompt,
        user_prompt=f"Evaluate relevance for query: {query}",
        output_format=Evaluation
    )

    return {"are_documents_relevant": evaluation.is_relevant}

def route_research(state: QuerySearchState, config: RunnableConfig) -> Literal["summarize_query_research", "web_research", "__end__"]:
    """ Route the research based on the documents relevance """

    if state["are_documents_relevant"]:
        return "summarize_query_research"
    elif config["configurable"].get("enable_web_search", False):
        return "web_research"
    else:
        print("Skipping query due to irrelevant documents and web search disabled.")
        return "summarize_query_research"

def web_research(state: QuerySearchState):
    print("--- Web research ---")
    output = tavily_search(state["query"])
    search_results = output["results"]

    return {"web_search_results": search_results}

def summarize_query_research(state: QuerySearchState):
    query = state["query"]

    retrieved_docs = state.get("retrieved_docs", [])

    if state.get("are_documents_relevant"):
        information = "\n\n".join(retrieved_docs)
    else:
        information = str(state.get("web_search_results", ""))

    summary_prompt = SUMMARIZER_PROMPT.format(
        query=query,
        documents=information
    )

    summary = invoke_ollama(
        model='phi3',
        system_prompt=summary_prompt,
        user_prompt=f"Generate summary for: {query}"
    )

    summary = parse_output(summary)["response"]

    return {
        "search_summaries": [summary],
        "retrieved_docs": retrieved_docs  #  keep passing forward
    }

def generate_final_answer(state: MedicalState, config: RunnableConfig):
    print("--- Generating final answer ---")

    answer_prompt = f"""
    Patient Symptoms: {state.get("symptoms")}
    Possible Conditions: {state.get("possible_conditions")}
    Severity Level: {state.get("severity_level")}

    Additional Information:
    {chr(10).join(state["search_summaries"])}

    Generate a medical report including:
    - Summary of condition
    - Possible diagnosis
    - Recommended tests
    - Precautions
    - When to see a doctor
    - Disclaimer: This is not a medical diagnosis
    """

    result = invoke_ollama(
        model='phi3',
        system_prompt="You are a clinical assistant.",
        user_prompt=answer_prompt
    )

    answer = parse_output(result)["response"]

    return {"final_answer": answer}

def emergency_check(state: MedicalState):
    print("--- Emergency check ---")

    if state["severity_level"] == "high" or len(state["risk_flags"]) > 0:
        return {
            "final_answer": "Emeregency!! Contact the doctor."
        }

    return {}

def run_query_subgraph(state: MedicalState):
    print("--- run_query_subgraph ---")

    subgraph = query_search_subgraph.compile()

    current_position = state.get("current_position", 0)
    queries = state.get("research_queries", [])

    # Reconstruct query safely
    query = None
    if current_position > 0 and current_position <= len(queries):
        query = queries[current_position - 1]

    if not query:
        return {
            **state,
            "debug_logs": state.get("debug_logs", []) + ["No query found, skipping"]
        }

    result = subgraph.invoke({
        "query": query
    })

    return {
        **state,
        "last_query": query,
        "search_summaries": state.get("search_summaries", []) + result.get("search_summaries", []),
        "retrieved_docs": result.get("retrieved_docs", []),  
        "debug_logs": state.get("debug_logs", []) + [f"Ran query: {query}"]
    }
# Create subghraph for searching each query
query_search_subgraph = StateGraph(QuerySearchState, input=QuerySearchStateInput, output=QuerySearchStateOutput)

# Define subgraph nodes for searching the query
query_search_subgraph.add_node(retrieve_rag_documents)
query_search_subgraph.add_node(evaluate_retrieved_documents)
query_search_subgraph.add_node(web_research)
query_search_subgraph.add_node(summarize_query_research)

# Set entry point and define transitions for the subgraph
query_search_subgraph.add_edge(START, "retrieve_rag_documents")
query_search_subgraph.add_edge("retrieve_rag_documents", "evaluate_retrieved_documents")
query_search_subgraph.add_conditional_edges("evaluate_retrieved_documents", route_research)
query_search_subgraph.add_edge("web_research", "summarize_query_research")
query_search_subgraph.add_edge("summarize_query_research", END)

# Create main graph
researcher_graph = StateGraph(
    MedicalState,
    input=MedicalStateInput,
    output=MedicalStateOutput,
    config_schema=Configuration
)

# Add nodes
researcher_graph.add_node(medical_reasoning)
researcher_graph.add_node(emergency_check)
researcher_graph.add_node(generate_medical_queries)
researcher_graph.add_node(search_queries)
researcher_graph.add_node("search_and_summarize_query", run_query_subgraph)
researcher_graph.add_node(generate_final_answer)

# Flow
researcher_graph.add_edge(START, "medical_reasoning")
researcher_graph.add_edge("medical_reasoning", "emergency_check")
researcher_graph.add_edge("emergency_check", "generate_medical_queries")
researcher_graph.add_edge("generate_medical_queries", "search_queries")

researcher_graph.add_edge("search_queries", "search_and_summarize_query")

researcher_graph.add_conditional_edges(
    "search_and_summarize_query",
    check_more_queries
)

researcher_graph.add_edge("generate_final_answer", END)
# Compile the researcher graph
researcher = researcher_graph.compile()