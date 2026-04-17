import os
from dataclasses import dataclass, fields
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

DEFAULT_MEDICAL_STRUCTURE = """
# Patient Summary
- Symptoms provided by the user
- Duration and severity (if mentioned)

# Possible Conditions (Differential Diagnosis)
- List of possible conditions
- Brief explanation for each

# Recommended Tests
- Suggested medical tests (if needed)

# Risk Assessment
- Low / Medium / High severity
- Red flag symptoms (if any)

# General Advice
- Basic precautions
- When to see a doctor

# Disclaimer
- This is not a medical diagnosis
- Consult a healthcare professional
"""

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    medical_structure: str = DEFAULT_MEDICAL_STRUCTURE

    max_search_queries: int = 5
    enable_web_search: bool = True

    # Medical AI controls
    enable_triage: bool = True  # detect emergencies
    enable_diagnosis: bool = True
    enable_test_recommendation: bool = True

    # ⚡ Model behavior
    temperature: float = 0.2
    model_name: str = "phi3"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        values: dict[str, Any] = {}

        for f in fields(cls):
            if not f.init:
                continue

            val = os.environ.get(f.name.upper(), configurable.get(f.name))

            # Type conversion
            if isinstance(f.default, bool) and isinstance(val, str):
                val = val.lower() == "true"
            elif isinstance(f.default, float) and val is not None:
                val = float(val)
            elif isinstance(f.default, int) and val is not None:
                val = int(val)

            if val is not None:
                values[f.name] = val

        return cls(**values)