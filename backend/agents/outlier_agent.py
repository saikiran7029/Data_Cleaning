import numpy as np
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class OutlierAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = """
You are an expert Data Engineer specializing in outlier detection. Analyze the statistical profile of each numeric column and decide on the best treatment strategy.

Based on the profile, generate a JSON object with a key "columns". Each object in the list should contain:
- "name": The column name.
- "suggested_action": One of "clip_to_bounds", "winsorize", "remove_outliers", "flag_outliers", or "skip".
- "reason": A brief justification.

Here is the data profile:
{input}

Return your response inside a single markdown code block as a JSON object.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        profile = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            stats = self.df[col].describe()
            profile.append({
                "name": col,
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "25%": float(stats["25%"]),
                "50%": float(stats["50%"]),
                "75%": float(stats["75%"]),
                "max": float(stats["max"])
            })
        return profile 