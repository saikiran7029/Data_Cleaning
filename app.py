import streamlit as st
import pandas as pd
import numpy as np
from backend.agents.root_agent import RootAgent
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Page Config & Styling ---
st.set_page_config(page_title="AI Data Cleaner", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background: linear-gradient(120deg, #f8fafc 0%, #e2e8f0 100%); color: #222; }
    section[data-testid="stSidebar"] { background: linear-gradient(120deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 0 20px 20px 0; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>AI Data Cleaning Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Effortlessly clean, standardize, and enhance your datasets with AI-powered agents.</p>", unsafe_allow_html=True)

# --- Helper Functions ---
def execute_code(df, code):
    df_copy = df.copy()
    try:
        exec(code, {'df': df_copy, 'pd': pd, 'np': np})
        return df_copy
    except Exception as e:
        st.error(f"Error executing code: {e}")
        return df

# --- UI Rendering Functions ---
def display_ui_for_agent(agent):
    agent_name = agent.__class__.__name__
    # Use a dictionary to map agent names to UI functions for cleaner code
    ui_function_map = {
        "DataTypeAgent": display_data_type_ui,
        "MissingValueAgent": display_missing_value_ui,
        "DuplicateAgent": display_duplicate_ui,
        "OutlierAgent": display_outlier_ui,
        "NormalizationAgent": display_normalization_ui,
        "ValueStandardizationAgent": display_value_standardization_ui,
        "FeatureGenerationAgent": display_feature_generation_ui,
        "ValidatingAgent": display_validation_results,
    }
    ui_function = ui_function_map.get(agent_name)
    if ui_function:
        ui_function(agent)

def display_data_type_ui(agent):
    if "actions_DataTypeAgent" not in st.session_state:
        st.session_state["actions_DataTypeAgent"] = agent.generate_actions()
    
    actions = st.session_state["actions_DataTypeAgent"]
    if not actions: return st.info("No data type suggestions available.")
    
    for col_action in actions:
        with st.expander(f"Column: {col_action['name']} (Current: {col_action['dtype']}) - {col_action.get('reason', '')}"):
            options = ["skip", "int64", "float64", "datetime64[ns]", "category", "string"]
            suggestion = col_action.get("suggested_dtype", "skip")
            idx = options.index(suggestion) if suggestion in options else 0
            user_dtype = st.selectbox("New data type:", options, index=idx, key=f"dtype_{col_action['name']}")
            st.session_state.user_choices[col_action['name']] = {"agent": agent, "choice": {"suggested_action": user_dtype, "reason": "User selected data type."}}

def display_missing_value_ui(agent):
    if "actions_MissingValueAgent" not in st.session_state:
        st.session_state["actions_MissingValueAgent"] = agent.generate_actions()

    actions = st.session_state["actions_MissingValueAgent"]
    if not actions: return st.info("No missing values found.")

    for col_action in actions:
        with st.expander(f"Column: {col_action['name']} - {col_action.get('reason', '')}"):
            options = ["skip", "drop_rows_with_missing_values", "drop_column", "fillna_mean", "fillna_median", "fillna_mode", "fillna_constant"]
            suggestion = col_action.get("suggested_action", "skip")
            idx = options.index(suggestion) if suggestion in options else 0
            user_action = st.radio("Action:", options, index=idx, key=f"mv_action_{col_action['name']}")
            
            choice = {"suggested_action": user_action, "reason": "User selected missing value treatment."}
            if user_action == "fillna_constant":
                val = st.text_input("Constant value:", key=f"mv_const_{col_action['name']}")
                choice["constant_value"] = val
            st.session_state.user_choices[col_action['name']] = {"agent": agent, "choice": choice}

# Add similar simplified UI functions for other agents
def display_duplicate_ui(agent):
    if "actions_DuplicateAgent" not in st.session_state:
        st.session_state["actions_DuplicateAgent"] = agent.generate_actions()

    actions = st.session_state["actions_DuplicateAgent"]
    if not actions or not actions[0].get("suggested_action"): return st.info("No duplicate rows found.")
    
    action = actions[0]
    st.write(f"**Suggestion:** {action.get('reason', '')}")
    options = ["skip", "drop_duplicates"]
    suggestion = action.get("suggested_action", "skip")
    idx = options.index(suggestion) if suggestion in options else 0
    user_action = st.radio("Action:", options, index=idx, key="dup_action")
    st.session_state.user_choices["duplicates"] = {"agent": agent, "choice": {"suggested_action": user_action, "reason": "User selected duplicate handling."}}

def display_outlier_ui(agent):
    actions = agent.generate_actions()
    st.session_state.actions = actions
    if not actions: return st.info("No numeric columns for outlier detection.")
    for col_action in actions:
        with st.expander(f"Column: {col_action['name']} - {col_action.get('reason', '')}"):
            options = ["skip", "clip_to_bounds", "winsorize", "remove_outliers", "flag_outliers"]
            suggestion = col_action.get("suggested_action", "skip")
            idx = options.index(suggestion) if suggestion in options else 0
            user_action = st.radio("Action:", options, index=idx, key=f"outlier_{col_action['name']}")
            st.session_state.user_choices[col_action['name']] = {"suggested_action": user_action, "reason": "User selected outlier treatment."}

def display_normalization_ui(agent):
    actions = agent.generate_actions()
    st.session_state.actions = actions
    if not actions: return st.info("No columns suitable for normalization.")
    for col_action in actions:
        with st.expander(f"Column: {col_action['name']} - {col_action.get('reason', '')}"):
            options = ["skip", "StandardScaler", "MinMaxScaler", "Log-Transform"]
            suggestion = col_action.get("suggested_strategy", "skip")
            idx = options.index(suggestion) if suggestion in options else 0
            user_strategy = st.radio("Strategy:", options, index=idx, key=f"norm_{col_action['name']}")
            st.session_state.user_choices[col_action['name']] = {"suggested_strategy": user_strategy, "reason": "User selected normalization strategy."}

def display_value_standardization_ui(agent):
    actions = agent.generate_actions()
    st.session_state.actions = actions
    if not actions: return st.info("No value standardization needed.")
    for col_action in actions:
        with st.expander(f"Column: {col_action['name']} - {col_action.get('reason', '')}"):
            st.write("Suggested Mappings:")
            st.json(col_action.get("mappings", []))
            st.session_state.user_choices[col_action['name']] = col_action

def display_feature_generation_ui(agent):
    actions = agent.generate_actions()
    st.session_state.actions = actions
    if not actions: return st.info("No new features suggested.")
    for feature in actions:
        with st.expander(f"Feature: {feature.get('name', '')} - {feature.get('reason', '')}"):
            st.code(feature.get('formula', ''), language="python")
            st.session_state.user_choices[feature.get('name', '')] = feature

def display_validation_results(agent):
    results = agent.generate_actions()
    if not results: return st.error("Validation agent failed to produce a result.")
    result = results[0]
    st.subheader("Final Validation")
    st.json(result)
    if result.get("status") == "completed":
        st.success("Data validation passed!")
    else:
        st.warning("Data validation found issues.")

def initialize_session_state():
    if "df" not in st.session_state: st.session_state.df = None
    if "root_agent" not in st.session_state: st.session_state.root_agent = None
    if "user_choices" not in st.session_state: st.session_state.user_choices = {}

def main():
    initialize_session_state()

    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.root_agent = RootAgent(df)
            st.session_state.user_choices = {}
            st.session_state.current_step = 0
            st.session_state.cleaning_logs = []
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    if st.session_state.df is not None:
        st.write("### Dataset Preview")
        st.dataframe(st.session_state.df.head())

        root_agent = st.session_state.root_agent
        cleaning_plan = root_agent.get_cleaning_plan()
        total_steps = len(cleaning_plan)
        step = st.session_state.get("current_step", 0)

        # If all steps are done, show final results
        if step >= total_steps:
            st.success("All cleaning steps completed!")
            st.write("### Cleaned Dataset")
            st.dataframe(st.session_state.df)
            st.write("### Full Cleaning Log (JSON)")
            st.json(st.session_state.cleaning_logs)
            return

        step_info = cleaning_plan[step]
        agent_name = step_info["agent_name"]
        st.header(f"Step {step+1} of {total_steps}: {agent_name}")
        st.write(step_info["reason"])

        agent = root_agent.get_agent(agent_name)
        if agent:
            display_ui_for_agent(agent)
        else:
            st.error(f"Could not find agent: {agent_name}")

        # Show cleaning log for this step if it exists
        if len(st.session_state.cleaning_logs) > step:
            st.subheader("Cleaning Log (JSON)")
            st.json(st.session_state.cleaning_logs[step])

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", disabled=step == 0):
                st.session_state.current_step = max(0, step - 1)
                st.rerun()
        with col2:
            if st.button("Apply Changes and Next", key=f"apply_next_{step}"):
                cleaned_df = st.session_state.df.copy()
                step_logs = []
                for column, data in st.session_state.user_choices.items():
                    # Only apply changes for columns relevant to this agent
                    if data.get("agent") != agent:
                        continue
                    choice = data["choice"]
                    action = choice.get("suggested_action") or choice.get("suggested_strategy") or "skip"
                    if action == "skip":
                        step_logs.append({
                            "column": column,
                            "user_choice": choice,
                            "code": None,
                            "status": "skipped"
                        })
                        continue
                    code_to_run = agent.generate_code_from_choice(column, choice)
                    log_entry = {
                        "column": column,
                        "user_choice": choice,
                        "code": code_to_run,
                        "status": None,
                        "error": None
                    }
                    if code_to_run and not code_to_run.startswith("#"):
                        try:
                            exec(code_to_run, {'df': cleaned_df, 'pd': pd, 'np': np})
                            log_entry["status"] = "success"
                        except Exception as e:
                            log_entry["status"] = "error"
                            log_entry["error"] = str(e)
                    else:
                        log_entry["status"] = "skipped"
                    step_logs.append(log_entry)
                # Save log for this step
                if len(st.session_state.cleaning_logs) > step:
                    st.session_state.cleaning_logs[step] = step_logs
                else:
                    st.session_state.cleaning_logs.append(step_logs)
                # Update df for next step
                st.session_state.df = cleaned_df
                # Move to next step
                st.session_state.current_step = step + 1
                st.rerun()

if __name__ == "__main__":
    main()