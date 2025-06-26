from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import json

class GeneralIssueAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = """
You are an expert Data Cleaner. You are given a description of a data quality issue. Suggest a fix and generate Python code to resolve it on a pandas DataFrame named 'df'.

**Issue Description:**
{input}

**Output Format (JSON only):**
{{
  "fix": "Short description of the fix.",
  "code": "Python code to fix the issue."
}}
Return your response inside a single markdown code block as a JSON object.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        return [{"issue": "User-described data quality issue"}]

    def _parse_llm_response(self, response: str, profile: list):
        """
        The response for this agent is a single JSON object, not a list.
        We wrap it in a list to conform to the base agent's expectation.
        """
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return [data] 
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing LLM response for {self.__class__.__name__}: {e}")
            return [{"fix": "Parsing Error", "code": "# Could not parse LLM response."}]