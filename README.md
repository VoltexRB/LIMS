# LIMS: LLM Interaction Manager

A Python module for managing interactions with large language models (LLMs).  
Supports use as a library in Python scripts and integration with JupyterLab notebooks.

## Installation
### From GitHub (public repository)

```bash
pip install git+https://github.com/VoltexRB/LIMS.git
```

### General usage

1. High-level API (recommended for most users).
This mode wraps the InteractionManager so you don’t need to deal with handlers directly.
```bash
import llm_interaction_manager.api as api

api.initialize()  # initializes the InteractionManager
api.start_conversation({"topic": "Test Conversation"})
api.set_rag_data({"my_data": "value"}, volatile=True)
api.response = send_prompt("Hello, LLM!")
print(response)
```
2. Direct InteractionManager (advanced usage).
This gives you full control over the LLM, persistent, and vector handlers.
```bash
from llm_interaction_manager.core.interaction_manager import InteractionManager

manager = InteractionManager()  # uses default handlers from settings
manager.start_conversation({"topic": "Advanced Test"})
response = manager.send_prompt("Hello, LLM!")
print(response)
```