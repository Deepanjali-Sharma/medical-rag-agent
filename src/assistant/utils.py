import os
import re
import shutil
from ollama import chat
from tavily import TavilyClient
from pydantic import BaseModel
from langchain_community.document_loaders import CSVLoader, TextLoader, PDFPlumberLoader
from src.assistant.vector_db import add_documents

DEBUG = True

class Evaluation(BaseModel):
    is_relevant: bool
    medical_relevance: bool  # is this medically useful?
    severity_level: str  # low / medium / high

class Queries(BaseModel):
    queries: list[str]  # search queries
    possible_conditions: list[str]  # diagnosis candidates
    recommended_tests: list[str]  # suggested tests

def parse_output(text):
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    output_match = re.search(r'</think>\s*(.*?)$', text, re.DOTALL)

    return {
        "reasoning": think_match.group(1).strip() if think_match else "",
        "response": output_match.group(1).strip() if output_match else text
    }

def format_documents_with_metadata(documents):
    formatted_docs = []
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown source')

        formatted_doc = (
            f"[MEDICAL SOURCE]\n"
            f"Source: {source}\n"
            f"Content: {doc.page_content}\n"
        )

        formatted_docs.append(formatted_doc)

    return "\n\n---\n\n".join(formatted_docs)

def invoke_ollama(model, system_prompt, user_prompt, output_format=None):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = chat(
        messages=messages,
        model=model,
        format=output_format.model_json_schema() if output_format else None,
        options={"temperature": 0.2}
    )
    content = response['message']['content']

    if output_format:
        return output_format.model_validate_json(content)
    return content
    
def invoke_llm(
    model,  # Specify the model name from OpenRouter
    system_prompt,
    user_prompt,
    output_format=None,
    temperature=0
):
        
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model, 
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base= "https://openrouter.ai/api/v1",
    )
    if DEBUG:
        print("\n[LLM INPUT]")
        print(user_prompt[:500])
    # If Response format is provided use structured output
    if output_format:
        llm = llm.with_structured_output(output_format)
    
    # Invoke LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm.invoke(messages)
    
    if output_format:
        return response
    return response.content # str response

def tavily_search(query, include_raw_content=True, max_results=3):
    """ Search the web using the Tavily API.

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""

    tavily_client = TavilyClient()
    return tavily_client.search(
        query + " medical",
        max_results=max_results,
        include_raw_content=include_raw_content
    )

def get_report_structures(reports_folder="report_structures"):
    """
    Loads report structures from .md or .txt files in the specified folder.
    Each file should be named as 'report_name.md' or 'report_name.txt' and contain the report structure.
    Returns a dictionary of report structures.
    """
    report_structures = {}

    # Create the folder if it doesn't exist
    os.makedirs(reports_folder, exist_ok=True)

    try:
        # List all .md and .txt files in the folder
        for filename in os.listdir(reports_folder):
            if filename.endswith(('.md', '.txt')):
                report_name = os.path.splitext(filename)[0]  # Remove extension
                file_path = os.path.join(reports_folder, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        report_structures[report_name] = {
                            "content": content
                        }
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

    except Exception as e:
        print(f"Error accessing reports folder: {str(e)}")

    return report_structures

def process_uploaded_files(uploaded_files):
    temp_folder = "temp_files"
    os.makedirs(temp_folder, exist_ok=True)

    try:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            temp_file_path = os.path.join(temp_folder, uploaded_file.name)

            # Save file temporarily
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Choose the appropriate loader
            if file_extension == "csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension in ["txt", "md"]:
                loader = TextLoader(temp_file_path)
            elif file_extension == "pdf":
                loader = PDFPlumberLoader(temp_file_path)
            else:
                continue

            # Load and append documents
            docs = loader.load()
            add_documents(docs)

        return True
    finally:
        # Remove the temp folder and its contents
        shutil.rmtree(temp_folder, ignore_errors=True)