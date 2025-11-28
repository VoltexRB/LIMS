from llm_interaction_manager.handlers.llm_handler_base import LLMHandlerBase
from langchain_together import *
from together import Together

class LangchainHandler(LLMHandlerBase):

    def __init__(self, data: dict = None):
        self.llm = None
        self.token = None
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

        :return: String "langchain"
        """
        return "langchain"

    def send_prompt(self, prompt: str, rag: list[str] =None) -> dict:
        """
        Sends the prompt with the optional RAG-Data to the external interface

        :param prompt: String-Object containing the Prompt
        :param rag: Optional RAG-Data to send
        :return: Dict containing prompt, response and possibly additional metadata
        """
        if self.llm is None:
            raise ValueError("LLM not initialized, use 'connect' first")

        full_prompt = prompt
        if rag:
            rag_text = "\n\n".join(rag)
            full_prompt = f"System Prompt: Use the following documents or previous conversation pieces to answer the question. Documents: \n{rag_text}\n\n Question: {prompt}\n"
        try:
            response = self.llm.invoke(full_prompt)
            content = getattr(response, "content", str(response))
            metadata = getattr(response, "response_metadata", None)
            result = {"response": content}
            if metadata:
                result["metadata"] = metadata
            return result
        except Exception as e:
            raise Exception(f"Error while trying to receive Answer from LLM_ {e}")

    def connect(self, data: dict) -> bool:
        """
        Connects to the external TogetherAI Service over Langchain

        :param data: Data to connect to the Service with, must contain at least "token" and "model"
        """
        if not {"model","token"} <= data.keys():
            raise ValueError("No Model defined or TogetherAI API token missing")

        self.token = data["token"]
        model = data["model"]
        if not self.validate_model_name(model):
            raise ValueError(f"Model '{model}' not found in TogetherAI")
        try:
            self.llm = ChatTogether(api_key=self.token, model=model)
        except Exception as e:
            raise Exception(f"Exception in runtime: {e}")
        self.auth = data
        return True

    def is_connected(self) -> bool:
        """
        Check if the service is connected, in which case the llm object will be initialized

        :return: True if the service is connected
        """
        if self.llm is None:
            return False
        return True

    def validate_model_name(self, model: str) -> bool:
        """
        Check if the model exists within TogetherAI.

        :param model: Model name to check against
        :return: True if the model exists
        """
        if self.token is None:
            raise ValueError("LLM not initialized, use 'connect' first")
        try:
            client = Together(api_key=self.token)
            models = client.models.list()
            model_ids = [m.id for m in models]  # extract all model IDs
            return model in model_ids
        except Exception as e:
            return False
