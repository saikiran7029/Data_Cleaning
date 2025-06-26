from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class DuplicateAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = """
You are a Data Cleaning Agent specializing in handling duplicate rows. Your job is to analyze the dataset and decide the best way to handle duplicates.
Based on the profile, generate a JSON object with a key "action".
- "suggested_action": Should be "drop_duplicates" or "skip".
- "reason": A brief justification for your choice.
Here is the data profile:
{input}
Return your response inside a single markdown code block as a JSON object.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        duplicate_rows = int(self.df.duplicated().sum())
        if duplicate_rows == 0:
            return []
        return [{
            "duplicate_rows": duplicate_rows,
            "total_rows": len(self.df),
        }]

    def generate_code_from_choice(self, choice: dict) -> str:
        action = choice.get("action")
        if not action or action == "skip":
            return "# No action chosen."
        if action == "drop_duplicates":
            return "df.drop_duplicates(inplace=True)"
        return "# No valid action chosen."