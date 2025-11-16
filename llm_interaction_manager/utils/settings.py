from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Union

class RAGMode(Enum):
    NONE = 0
    PERSISTENT = 1
    VOLATILE = 2
    DYNAMIC = 3

@dataclass
class Settings:
    """
    Defaults or fallbacks if keys not found:
    - handlers: {}                 # No handler configurations have been defined yet
    - default_handlers: {}         # No default handler selections made
    - use_rag_data: RAGMode.NONE   # No RAG mode selected
    - on_the_fly_data: {}          # No on-the-fly data paths or keys defined
    - default_rag_data: {}         # No default RAG data configured
    - default_system_prompt: "-1"  # Default prompt not yet specified
    - wait_for_manual_data: False   # Default is bypassing comments
    - default_export_path: "-1"    # Export path not yet set by the user
    """
    handlers: Dict[str, dict] = field(default_factory=dict)  # key = handler name, value = its config
    default_handlers: Dict[str, str] = field(default_factory=dict)  # e.g., {"llm": "huggingface"}
    use_rag_data: RAGMode = RAGMode.NONE
    on_the_fly_data: Union[Dict[str, str], List[str]] = field(default_factory=dict)
    default_rag_data: Union[Dict[str, str], List[str]] = field(default_factory=dict)
    default_system_prompt: str = "-1"
    wait_for_manual_data: bool = False
    default_export_path: str = "-1"
    send_conversation_history: bool = False
