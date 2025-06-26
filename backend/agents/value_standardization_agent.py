import json
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class ValueStandardizationAgent(BaseAgent):
    """
    An agent that specializes in standardizing categorical values in a dataframe.
    """
    def _get_prompt_template(self) -> ChatPromptTemplate:
        # This prompt is kept identical to the original one.
        prompt = """
You are a Data Cleaning Agent specializing in value standardization. Your job is to analyze the dataset profile and suggest the best way to standardize values for each categorical column.

The profile contains a list of columns with their unique values. For each column, decide if standardization is needed. For example, 'USA' and 'U.S.A' should be mapped to a single value.

Based on the profile, generate a JSON object with a key "columns". This key should contain a list of objects, where each object represents a column and has the following structure:
- "name": The name of the column.
- "suggested_action": Should be "standardize_values". If no action is needed, use "skip".
- "reason": A brief explanation of why this action is suggested.
- "mappings": A list of mapping objects, e.g., [{"from": "U.S.A", "to": "USA"}]. This should be empty if the action is "skip".

Here is the data profile:
{input}

Return your response inside a single markdown code block as a JSON object.
Example:
```json
{{
  "columns": [
    {{
      "name": "country",
      "suggested_action": "standardize_values",
      "reason": "Inconsistent country names found.",
      "mappings": [
        {{"from": "U.S.A", "to": "USA"}},
        {{"from": "United States", "to": "USA"}}
      ]
    }},
    {{
      "name": "status",
      "suggested_action": "skip",
      "reason": "Values are already consistent.",
      "mappings": []
    }}
  ]
}}
```
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        """Profiles string/object columns and collects their unique values."""
        profile = []
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            unique_values = list(self.df[col].dropna().unique())
            if len(unique_values) > 1:
                profile.append({
                    "name": col,
                    "unique_values": unique_values[:30],  # limit for prompt size
                    "num_unique": len(unique_values)
                })
        return profile 