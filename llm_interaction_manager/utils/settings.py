from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Union

class ContextMode(Enum):
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
    - use_context_data: ContextMode.NONE   # No Context mode selected
    - on_the_fly_data: {}          # No on-the-fly data paths or keys defined
    - default_context_data: {}         # No default Context data configured
    - system_prompt: "-1"  # Default prompt not yet specified
    - wait_for_manual_data: False   # Default is bypassing comments
    - default_export_path: "-1"    # Export path not yet set by the user
    """
    handlers: Dict[str, dict] = field(default_factory=dict)  # key = handler name, value = its config
    default_handlers: Dict[str, str] = field(default_factory=dict)  # e.g., {"llm": "huggingface"}
    use_context_data: ContextMode = ContextMode.NONE
    on_the_fly_data: Union[Dict[str, str], List[str]] = field(default_factory=dict)
    default_context_data: Union[Dict[str, str], List[str]] = field(default_factory=dict)
    system_prompt: str = "-1"
    wait_for_manual_data: bool = False
    default_export_path: str = "-1"
    send_conversation_history: bool = False
