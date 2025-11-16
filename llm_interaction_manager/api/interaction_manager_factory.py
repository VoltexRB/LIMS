from enum import Enum
import importlib
import inspect
from llm_interaction_manager.core.interaction_manager import InteractionManager
from llm_interaction_manager.utils.settings_handler import SettingsHandler, SettingsSection

# Hier können weitere Enums hinzugefügt werden, wenn weitere Klassen für weitere Schnittstellen implementiert werden.
# Wichtig: Wenn zwei Enums den gleichen Namen haben, geht die Software davon aus, dass sie alle Funktionalitäten abdecken.
# Zum Beispiel: Postgres wird aufgeführt in Persistent und Vector. Postgres muss also beide Funktionen unterstützen und beide abstrakten Basisklassen implementieren

class LLMEnum(str, Enum):
    LANGCHAIN="langchain"
    HUGGINGFACE="huggingface"
    SETTINGS="-1"

class PersistentEnum(str, Enum):
    POSTGRES = "postgres"
    MONGODB = "mongodb"
    SETTINGS="-1"

class VectorEnum(str, Enum):
    POSTGRES = "postgres"
    CHROMADB = "chromadb"
    SETTINGS="-1"

def _load_handler(handler_key: str):
    """
    Lädt automatisch eine Handler-Klasse aus handlers/<handler_key>_handler.py
    :param handler_key: Value eines Enum aus einem der drei HandlerEnums
    :returns
    """
    module_name = f"{handler_key.lower()}_handler"
    class_name = f"{handler_key.title()}Handler"
    module_path = f"llm_interaction_manager.handlers.{module_name}"
    module = importlib.import_module(module_path)

    # Suche die Klasse im Modul
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if name == class_name:
            return cls()

    raise ValueError(f"Class {class_name} not found in {module_name}.py")

def initialize(llm: LLMEnum = LLMEnum.SETTINGS, persistent: PersistentEnum = PersistentEnum.SETTINGS, vector: VectorEnum = VectorEnum.SETTINGS) -> InteractionManager:
    """
    Erstellt einen neuen InteractionManager mit den angegebenen Ressourcen.
    Wenn keine Schnittstellen angegeben werden oder SETTINGS bei den Enums ausgewählt werden,
    wird die Standardeinstellung verwendet, falls vorhanden.
    :return: Einen InteractionManager mit den Schnittstellenklassen
    """
    # --- LLM ---
    if llm == LLMEnum.SETTINGS:
        llm_dict = SettingsHandler.read_setting(SettingsSection.DEFAULT_HANDLERS, "llm")
        if not llm_dict or "value" not in llm_dict or not llm_dict["value"]:
            raise ValueError("Could not find previously set value for llm selection.")
        handler_key = llm_dict["value"]
    else:
        handler_key = llm.value

    llm_handler = _load_handler(handler_key)

    # --- Persistent ---
    if persistent == PersistentEnum.SETTINGS:
        persistent_dict = SettingsHandler.read_setting(SettingsSection.DEFAULT_HANDLERS, "persistent")
        if not persistent_dict or "value" not in persistent_dict or not persistent_dict["value"]:
            raise ValueError("Could not find previously set value for persistent storage selection.")
        persistent_key = persistent_dict["value"]
    else:
        persistent_key = persistent.value

    # --- Vector ---
    if vector == VectorEnum.SETTINGS:
        vector_dict = SettingsHandler.read_setting(SettingsSection.DEFAULT_HANDLERS, "vector")
        if not vector_dict or "value" not in vector_dict or not vector_dict["value"]:
            raise ValueError("Could not find previously set value for vector storage selection.")
        vector_key = vector_dict["value"]
    else:
        vector_key = vector.value

    # --- Handler laden ---
    persistent_handler = _load_handler(persistent_key)
    if persistent_key == vector_key:
        vector_handler = persistent_handler
    else:
        vector_handler = _load_handler(vector_key)

    return InteractionManager(llm_handler, persistent_handler, vector_handler)



