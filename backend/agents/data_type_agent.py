import pandas as pd
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import re
import json

class DataTypeAgent(BaseAgent):
    """
    An agent that specializes in analyzing and correcting data types.
    """
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = """
You are a Data Cleaning Agent specializing in analyzing and correcting data types. Your job is to analyze the dataset and suggest the best data type for each column.

Based on the profile, generate a JSON object with a key "columns". This key should contain a list of objects, where each object represents a column and has the following structure:
- "name": The name of the column.
- "suggested_dtype": The suggested data type (e.g., "int64", "float64", "datetime64[ns]", "category", "string", or "skip").
- "reason": A brief justification for your choice.

Here is the data profile:
{input}

Return your response inside a single markdown code block as a JSON object.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        """Profiles columns for data type analysis."""
        profile = []
        for col in self.df.columns:
            unique_samples = self.df[col].dropna().unique()
            sample_size = min(5, len(unique_samples))
            samples = pd.Series(unique_samples).sample(sample_size).tolist() if sample_size > 0 else []

            profile.append({
                "name": col,
                "dtype": str(self.df[col].dtype),
                "sample_values": [str(s) for s in samples]
            })
        return profile

    def _parse_llm_response(self, response: str, profile: list):
        """Adds the original dtype to the parsed response for UI display."""
        actions = super()._parse_llm_response(response, profile)
        for col_profile, llm_action in zip(profile, actions):
            llm_action['dtype'] = col_profile['dtype']
        return actions 