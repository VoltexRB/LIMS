from pathlib import Path
import pandas as pd
import json
from llm_interaction_manager.api import lims_interface as api
from llm_interaction_manager.api.interaction_manager_factory import LLMEnum, PersistentEnum, VectorEnum
from llm_interaction_manager.core.interaction_manager import ConnectionType
from llm_interaction_manager.utils import ContextMode

api.initialize(llm=LLMEnum.LANGCHAIN, vector=VectorEnum.CHROMADB, persistent=PersistentEnum.MONGODB)

# Connection data
config_path = Path(__file__).parent / "test_config.json"
with config_path.open("r", encoding="utf-8") as file:
    config = json.load(file)
llm_config = config["handlers"]["langchain"]
llm_data = {
    "model": llm_config["model"],
    "token": llm_config["token"]
}
vector_data = {
    "client_type": "PERSISTENT",
    "persistent_client_db_path": "D:/chroma"
}
persistent_data = {
    "host": "localhost",
    "port": 27017,
    "database": "nutzungskontext"
}

api.connect(ConnectionType.LLM, llm_data)
api.connect(ConnectionType.VECTOR, vector_data)
api.connect(ConnectionType.PERSISTENT, persistent_data)

print(api.is_connected(ConnectionType.LLM))
print(api.is_connected(ConnectionType.PERSISTENT))
print(api.is_connected(ConnectionType.VECTOR))

api.start_conversation()

DIR = Path(__file__).parent
excel_sheet = DIR / "Nutzungstest Prompts.xlsx"
context_file = DIR / "roentgen_dta.txt"

prompts = pd.read_excel(excel_sheet,sheet_name="example_prompts_evaluation")

with context_file.open("r", encoding="utf-8") as file:
    roentgen_data = file.read()

marker = "EINE NEUE ART\nVON\nSTRAHLEN."

start = roentgen_data.find(marker)

if start == -1:
    raise ValueError("Could not find start of Röntgen document")

roentgen_data = roentgen_data[start:]
results = []

for _, row in prompts.iterrows():
    prompt = row["prompt"]
    context_source = row["context_source"]

    #keine weiteren Kontextdaten
    if context_source == "none":
        api.delete_context_data()

    #Dokument als Kontext
    elif context_source == "roentgen":
        api.set_context_data({"text":roentgen_data}, volatile=True)

    #dynamische RAG aus den Vektordaten
    elif context_source == "chromadb":
        api.set_context_mode(ContextMode.DYNAMIC)

    response = api.send_prompt(prompt)

    results.append({
        "id": int(row["id"]),
        "response": response["content"]
    })

    print("Prompt: " + prompt)
    print("Response: " + response["content"])
    print()

results_file = DIR / "Nutzungstest Ergebnisse.json"

with open(results_file, "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)