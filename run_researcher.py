from src.assistant.graph import researcher
from src.assistant.vector_db import get_or_create_vector_db
from dotenv import load_dotenv

load_dotenv()

report_structure = """
1. Introduction
- Brief overview of the research topic or question.
- Purpose and scope of the report.

2. Main Body
- For each section (e.g., Section 1, Section 2, Section 3, etc.):
  - Provide a subheading related to the key aspect of the research.
  - Include explanation, findings, and supporting details.

3. Key Takeaways
- Bullet points summarizing the most important insights or findings.

4. Conclusion
- Final summary of the research.
- Implications or relevance of the findings.
"""

# Define the initial state
user_query = input("Enter your symptoms: ")

initial_state = {
    "user_instructions": user_query,
}

# Langgraph researcher config
config = {
  "configurable": {
    "enable_web_search": True,
    "max_search_queries": 5,

    # Medical settings
    "enable_triage": True,
    "enable_diagnosis": True,
    "enable_test_recommendation": True,

    "temperature": 0.2,
    "model_name": "llama3"
}}
# Init vector store
# Must add your own documents in the /files directory before running this script
vector_db = get_or_create_vector_db()

# Run the researcher graph
for output in researcher.stream(initial_state, config=config):
    for key, value in output.items():
        print(f"\n Finished: {key}")

        if isinstance(value, dict):
            for k, v in value.items():
                print(f"\n{k.upper()}:\n{v}")
        else:
            print(value)

