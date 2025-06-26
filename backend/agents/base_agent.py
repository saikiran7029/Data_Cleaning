import json
import re
import pandas as pd
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from utils.openai_client import llm

class BaseAgent:
    """
    A base class for all data cleaning agents, powered by LangChain.
    Each agent profiles data, suggests actions, and can generate code for those actions.
    """
    def __init__(self, dataframe: pd.DataFrame, all_agents=None):
        self.df = dataframe
        self.llm = llm # Use the centralized LLM
        
        # Each subclass must define its own prompt for suggesting actions
        action_prompt_template = self._get_prompt_template()
        
        # Create a simple chain for invoking the LLM, not a complex agent executor
        if action_prompt_template:
            self.action_chain: Runnable = action_prompt_template | self.llm
        else:
            self.action_chain = None

    def _get_prompt_template(self) -> ChatPromptTemplate:
        """
        Subclasses must implement this to provide their specific system prompt
        for **suggesting cleaning actions**.
        """
        raise NotImplementedError("Each agent must provide a prompt template for suggesting actions.")

    def _get_code_generation_prompt_template(self) -> ChatPromptTemplate:
        """
        Provides a standardized prompt for **generating Python code** based on a suggested action.
        """
        prompt = """
You are a Python code generation assistant. Your task is to write a single, executable line of pandas code to perform a specific data cleaning action on a DataFrame named 'df'.

- **DataFrame**: A pandas DataFrame named `df` is pre-defined and available.
- **Column**: You are working on the column named `{column_name}`.
- **Action**: The suggested action is: `{action_details}`
- **Reason**: The reason for this action is: `{reason}`

Based on this, generate a single line of python code. Do not add comments, explanations, or markdown. Output only the raw code.

Example for a 'fillna_median' action on column 'age':
df['age'].fillna(df['age'].median(), inplace=True)
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        """
        Subclasses must implement this to generate a profile of the dataframe
        relevant to their specific task.
        """
        raise NotImplementedError

    def generate_actions(self):
        """
        The main entry point for an agent to suggest cleaning actions.
        It profiles the data, invokes the LangChain chain, and parses the result.
        """
        if not self.action_chain:
            return []

        profile = self.profile_columns()
        if not profile:
            return []
        
        profile_info = json.dumps(profile, indent=2, default=str)
        
        try:
            # Invoke the simple chain, not an agent executor
            response = self.action_chain.invoke({"input": profile_info})
            # The output of an LLM call is in the 'content' attribute
            output = response.content
            return self._parse_llm_response(output, profile)
        except Exception as e:
            print(f"Error invoking agent for {self.__class__.__name__}: {e}")
            return self._create_error_response(profile)

    def _parse_llm_response(self, response: str, profile: list):
        """
        Parses the JSON response from the LLM. Subclasses can override this.
        """
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            actions = data.get("columns", [])
            
            # Fallback for agents that might return a single action dict
            if not actions and isinstance(data, dict):
                return [data]

            for i, col_action in enumerate(actions):
                if i < len(profile):
                    col_action.setdefault("name", profile[i].get("name"))
                col_action.setdefault("reason", "No reason provided by LLM.")
            return actions
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing LLM response for {self.__class__.__name__}: {e}")
            return self._create_error_response(profile)

    def _extract_json(self, response: str) -> str:
        """Extracts a JSON string from a markdown code block."""
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def _create_error_response(self, profile: list) -> list:
        """Creates a default 'skip' action when an error occurs."""
        if not profile:
            return [{"name": "unknown", "suggested_action": "skip", "reason": "Agent failed due to an error and no profile was available."}]
            
        return [
            {
                "name": col.get("name"),
                "suggested_action": "skip",
                "reason": "Agent failed to generate a valid suggestion due to an internal error."
            }
            for col in profile
        ]

    def generate_code_from_choice(self, column_name: str, choice: dict) -> str:
        """
        Generates Python code for a chosen cleaning action by invoking the LLM.
        """
        action = choice.get("suggested_action") or choice.get("action") or choice.get("suggested_strategy")
        if not action or action == "skip":
            return "# No action chosen."
            
        # For feature generation, the formula is the code. No need to call LLM.
        if "formula" in choice:
            return choice["formula"]
        
        # For general issues, the code is already generated.
        if "code" in choice:
            return choice["code"]

        action_details = f"Action: {action}"
        if choice.get("constant_value") is not None:
            action_details += f", Constant Value: '{choice['constant_value']}'"
        
        reason = choice.get("reason", "No reason provided.")

        code_gen_prompt = self._get_code_generation_prompt_template()
        code_gen_chain = code_gen_prompt | self.llm
        
        response = code_gen_chain.invoke({
            "column_name": column_name,
            "action_details": action_details,
            "reason": reason,
        })
        
        code = response.content.strip()
        cleaned_code = re.sub(r"^```python\n|```$", "", code).strip()
        return cleaned_code