from llm_interaction_manager.handlers.llm_handler_base import LLMHandlerBase
from transformers import *
from huggingface_hub import list_models, login
import huggingface_hub, logging, transformers

class HuggingfaceHandler(LLMHandlerBase):

    def __init__(self, data: dict = None):
        self.llm = None
        if data is not None:
            self.connect(data)

    def get_info(self) -> dict:
        """
        Returns connection-data to be saved in the Settings-Object

        :return: Dict containing authentication-data
        """
        return self.auth

    def get_name(self) -> str:
        """
        Returns the name of the Handler for dynamic binding

        :return: String "huggingface"
        """
        return "huggingface"

    def is_connected(self) -> bool:
        """
        As the model is run locally, only check if it is instantiated

        :return: true if model is instantiated
        """
        return self.llm is not None

    def connect(self, data: dict):
        """
        As the model is run locally, create llm and pipeline objects to use for further usage

        :param data: Data to instantiate the llm pipeline with. must contain at least "token" and "model"
        """
        if "model" not in data:
            raise ValueError("No Model defined")
        if "token" in data:
            login(token=data["token"])
        if not self.validate_model_name(data["model"]): return False
        try:
            transformers.logging.set_verbosity_error()
            huggingface_hub.logging.set_verbosity_error()
            self.llm = pipeline("text-generation", model=data["model"])
            self.auth = data

        except Exception as e:
            raise RuntimeError("The model you are trying to access might be private or restricted. Please provide an access token in your connection data")

    def send_prompt(self, prompt: str, rag: list[str] =None) -> dict:
        """
        Sends the prompt with the optional RAG-Data to the external interface

        :param prompt: String-Object containing the Prompt
        :param rag: Optional RAG-Data to send
        :return: Dict containing prompt, response and possibly additional metadata
        """
        if self.llm is None:
            raise ValueError("No LLM initiated. Use 'connect' first")

        # Merge RAG documents into the prompt if provided
        full_prompt = prompt
        if rag:
            rag_text = "\n\n".join(rag)  # combine snippets
            full_prompt = f"System Prompt: Use the following documents or previous conversation pieces to answer the question. Documents: \n{rag_text}\n\n Question: {prompt}\n"

        # Call the LLM pipeline
        result = self.llm(full_prompt)
        generated = result[0]["generated_text"]
        if generated.startswith(full_prompt):
            generated = generated[len(full_prompt):]  # cut off the prompt
        return {"response": generated, "prompt": full_prompt}

    def validate_model_name(self, model: str) -> bool:
        """
        Check if the model exists within huggingface. If not, prints a list of 5 close model names

        :param model: Model name to check against
        :return: True if the model exists
        """
        model_names = list(list_models(search=model, limit=5))
        found_names = [m.id for m in model_names]
        if found_names[0].strip() == model:
            return True
        else:
            print("Invalid Model")
            print("Valid Model names:" + "\n - ".join(found_names))
            return False

