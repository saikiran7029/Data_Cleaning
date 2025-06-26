class AgentMemory:
    """
    Tracks all steps, actions, reasons, and user overrides for a data cleaning session.
    """
    def __init__(self):
        self.steps = []  # List of dicts: {step, agent, action, reason, user_override, status, code}

    def log_step(self, step_info):
        self.steps.append(step_info)

    def get_log(self):
        return self.steps

    def clear(self):
        self.steps = []

    def last(self):
        return self.steps[-1] if self.steps else None 