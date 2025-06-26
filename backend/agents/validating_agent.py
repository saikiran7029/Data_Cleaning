from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import json

class ValidatingAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = """
You are an expert Data Quality Validator. Analyze the dataset profile below and list all data quality issues found.
If no issues are found, return an empty "issues" list.

**Dataset Profile:**
{input}

**Output Format (JSON only):**
{{
  "status": "completed" or "issues_found",
  "issues": [
    {{
      "description": "description of the issue",
      "severity": "high/medium/low"
    }}
  ]
}}
Return your response inside a single markdown code block as a JSON object.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        return [{
            "missing_values": self.df.isnull().sum().sum(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "data_types": self.df.dtypes.to_string(),
        }]