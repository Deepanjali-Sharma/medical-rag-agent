import pyperclip
import streamlit as st
import streamlit_nested_layout
from src.assistant.graph import researcher
from src.assistant.utils import get_report_structures, process_uploaded_files
from dotenv import load_dotenv

load_dotenv()

def generate_response(user_input, enable_web_search, report_structure, max_search_queries):
    """
    Generate response using the researcher agent and stream steps
    """
    # Initialize state for the researcher
    initial_state = {
        "user_instructions": user_input,
    }
    
    # Langgraph researcher config
    config = {"configurable": {
        "enable_web_search": enable_web_search,
        "medical_structure": report_structure,
        "max_search_queries": max_search_queries,
    }}

    # Create the status for the global "Researcher" process
    langgraph_status = st.status("**Researcher Running...**", state="running")

    # Force order of expanders by creating them before iteration
    with langgraph_status:
        process_expander = st.expander("Reasoning & Query Generation", expanded=False)
        search_expander = st.expander("Search & RAG Analysis", expanded=True)

        steps = []

        # Run the researcher graph and stream outputs
        for output in researcher.stream(initial_state, config=config):
            for key, value in output.items():
                # 1. Capture Medical Reasoning & Query Gen
                if key in ["medical_reasoning", "generate_medical_queries"]:
                    with process_expander:
                        st.write(f"**Step: {key}**")
                        st.json(value)

                # 2. Capture the Subgraph results (Search & Summarize)
                elif key == "search_and_summarize_query":
                    with search_expander:
                        if "last_query" in value:
                            st.write(f"###  Query: {value['last_query']}")

                        if "retrieved_docs" in value:
                            st.write("### Retrieved Docs")
                            for doc in value["retrieved_docs"]:
                                st.write(doc[:200] + "...")

                        if "search_summaries" in value:
                            st.write("###  Summary")
                            summaries = value.get("search_summaries", [])

                            if summaries:
                                st.write("### Summary")
                                st.write(summaries[-1])
                            else:
                                st.write(" No summary available for this query")

                        if "debug_logs" in value:
                            with st.expander("Debug Logs"):
                                for log in value["debug_logs"]:
                                    st.write(log)

                # 3. Capture Final Answer
                elif key == "generate_final_answer":
                    steps.append(value)

                steps.append({"step": key, "content": value})

    langgraph_status.update(state="complete", label="Research completed!")
    return steps[-1] if steps else {"final_answer": "No response generated."}

def clear_chat():
    st.session_state.messages = []
    st.session_state.processing_complete = False
    st.session_state.uploader_key = 0

def main():
    st.set_page_config(page_title="DeepSeek RAG Researcher", layout="wide")

    # Initialize session states
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_report_structure" not in st.session_state:
        st.session_state.selected_report_structure = None
    if "max_search_queries" not in st.session_state:
        st.session_state.max_search_queries = 5  # Default value of 5
    if "files_ready" not in st.session_state:
        st.session_state.files_ready = False  # Tracks if files are uploaded but not processed

    # Title row with clear button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Medical assistant")
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()

    # Sidebar configuration
    st.sidebar.title("Research Settings")

    # Add report structure selector to sidebar
    report_structures = get_report_structures()
    default_report = "standard report"

    # Get available structures
    report_options = list(report_structures.keys())

    if not report_options:
        # Fallback if no .txt files are found
        st.error("No report templates (.txt) found in report_structures folder.")
        st.stop()

    # Safe index lookup
    default_report = "standard report"
    try:
        default_index = list(map(str.lower, report_options)).index(default_report.lower())
    except (ValueError, IndexError):
        default_index = 0

    selected_structure = st.sidebar.selectbox(
        "Select Report Structure",
        options=report_options,
        index=default_index
    )
    st.session_state.selected_report_structure = report_structures[selected_structure]

    # Maximum search queries input
    st.session_state.max_search_queries = st.sidebar.number_input(
        "Max Number of Search Queries",
        min_value=1,
        max_value=10,
        value=st.session_state.max_search_queries,
        help="Set the maximum number of search queries to be made. (1-10)"
    )
    
    enable_web_search = st.sidebar.checkbox("Enable Web Search", value=False)

    # Upload file logic
    uploaded_files = st.sidebar.file_uploader(
        "Upload New Documents",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # Check if files are uploaded but not yet processed
    if uploaded_files:
        st.session_state.files_ready = True  # Mark that files are available
        st.session_state.processing_complete = False  # Reset processing status

    # Display the "Process Files" button **only if files are uploaded but not processed**
    if st.session_state.files_ready and not st.session_state.processing_complete:
        process_button_placeholder = st.sidebar.empty()  # Placeholder for dynamic updates

        with process_button_placeholder.container():
            process_clicked = st.button("Process Files", use_container_width=True)

        if process_clicked:
            with process_button_placeholder:
                with st.status("Processing files...", expanded=False) as status:
                    # Process files (Replace this with your actual function)
                    if process_uploaded_files(uploaded_files):
                        st.session_state.processing_complete = True
                        st.session_state.files_ready = False  # Reset files ready flag
                        st.session_state.uploader_key += 1  # Reset uploader to allow new uploads

                    status.update(label="Files processed successfully!", state="complete", expanded=False)
                    # st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])  # Show the message normally

            # Show copy button only for AI messages at the bottom
            if message["role"] == "assistant":
                if st.button("📋", key=f"copy_{len(st.session_state.messages)}"):
                    pyperclip.copy(message["content"])

    # Chat input and response handling
    if user_input := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate and display assistant response
        report_structure = st.session_state.selected_report_structure["content"]
        assistant_response = generate_response(
            user_input, 
            enable_web_search, 
            report_structure,
            st.session_state.max_search_queries
        )

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": assistant_response["final_answer"]})

        with st.chat_message("assistant"):
            st.write(assistant_response["final_answer"])  # AI response

            # Copy button below the AI message
            if st.button("📋", key=f"copy_{len(st.session_state.messages)}"):
                pyperclip.copy(assistant_response["final_answer"])

if __name__ == "__main__":
    main()