from backend.agents.missing_value_agent import MissingValueAgent
from backend.agents.outlier_agent import OutlierAgent
from backend.agents.duplicate_agent import DuplicateAgent
from backend.agents.data_type_agent import DataTypeAgent
from backend.agents.normalization_agent import NormalizationAgent
from backend.agents.value_standardization_agent import ValueStandardizationAgent
from backend.agents.feature_generation_agent import FeatureGenerationAgent
from backend.agents.validating_agent import ValidatingAgent
from backend.agents.general_issue_agent import GeneralIssueAgent

class RootAgent:
    """
    The main orchestrator that holds instances of all specialized cleaning agents.
    """
    def __init__(self, df):
        self.df = df
        
        agent_classes = {
            "Data Types": DataTypeAgent,
            "Missing Values": MissingValueAgent,
            "Duplicates": DuplicateAgent,
            "Outliers": OutlierAgent,
            "Normalization": NormalizationAgent,
            "Value Standardization": ValueStandardizationAgent,
            "Feature Generation": FeatureGenerationAgent,
            "Validation": ValidatingAgent,
            "General Issue": GeneralIssueAgent,
        }
        
        self.agents = {
            name: cls(self.df) for name, cls in agent_classes.items()
        }

    def get_agent(self, agent_name: str):
        """Returns an instance of the requested agent."""
        return self.agents.get(agent_name)

    def get_cleaning_plan(self):
        """Returns a predefined, static cleaning plan for the UI."""
        return [
            {"agent_name": "Data Types", "reason": "Analyze and fix column data types."},
            {"agent_name": "Missing Values", "reason": "Find and handle missing values."},
            {"agent_name": "Duplicates", "reason": "Find and remove duplicate rows."},
            {"agent_name": "Outliers", "reason": "Identify and treat outlier values."},
            {"agent_name": "Value Standardization", "reason": "Standardize inconsistent categorical values."},
            {"agent_name": "Normalization", "reason": "Normalize numeric columns for modeling."},
            {"agent_name": "Feature Generation", "reason": "Generate new features from existing data."},
            {"agent_name": "Validation", "reason": "Perform a final validation of the data quality."}
        ]