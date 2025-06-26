import numpy as np
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class NormalizationAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = """
You are a Data Cleaning Agent specializing in normalization. Your job is to analyze the dataset and suggest the best normalization strategy for each numeric column.

Based on the profile, generate a JSON object with a key "columns". Each object in the list should contain:
- "name": The column name.
- "suggested_strategy": One of "StandardScaler", "MinMaxScaler", "Log-Transform", or "skip".
- "reason": A brief justification for your choice.

Here is the data profile:
{input}

Return your response inside a single markdown code block as a JSON object.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        profile = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].nunique() <= 10 or 'id' in col.lower():
                continue
            profile.append({
                "name": col,
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "skew": float(self.df[col].skew()),
            })
        return profile 