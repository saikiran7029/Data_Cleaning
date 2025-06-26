from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import json

class FeatureGenerationAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = """
You are a Data Cleaning Agent specializing in feature generation. Your job is to analyze the dataset and suggest new, valuable features that could be generated from existing columns.

Based on the profile, generate a JSON object with a key "features". Each object in the list should contain:
- "name": The name of the new feature column.
- "formula": The pandas code/formula to generate the feature.
- "reason": A brief justification for why this feature is useful.

Here is the data profile:
{input}

Return your response inside a single markdown code block as a JSON object.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        return [{
            "name": col,
            "dtype": str(self.df[col].dtype)
        } for col in self.df.columns]

    def _parse_llm_response(self, response: str, profile: list):
        """Override to parse a 'features' key instead of 'columns'."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            actions = data.get("features", [])
            
            for action in actions:
                action.setdefault("name", "unnamed_feature")
                action.setdefault("formula", "")
                action.setdefault("reason", "No reason provided.")
            
            return actions
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing LLM response for {self.__class__.__name__}: {e}")
            return []