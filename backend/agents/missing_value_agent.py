from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class MissingValueAgent(BaseAgent):
    """
    An agent that specializes in detecting and suggesting treatments for missing values.
    """
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = """
You are a Data Cleaning Agent specializing in handling missing values. Your job is to analyze the dataset and decide the most appropriate missing value treatment per column.

Based on the profile, generate a JSON object with a key "columns". This key should contain a list of objects, where each object represents a column and has the following structure:
- "name": The name of the column.
- "suggested_action": Choose one of: "drop_column", "drop_rows_with_missing_values", "fillna_mean", "fillna_median", "fillna_mode", "fillna_constant", or "skip".
- "constant_value": If using "fillna_constant", specify the actual constant.
- "reason": A clear, concise explanation for the decision.

Here is the data profile:
{input}

Return your response inside a single markdown code block as a JSON object.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        """Profiles columns for missing value analysis."""
        profile = []
        for col in self.df.columns:
            if self.df[col].isnull().any():
                missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                profile.append({
                    "name": col,
                    "dtype": str(self.df[col].dtype),
                    "missing_pct": round(missing_pct, 2),
                })
        return profile