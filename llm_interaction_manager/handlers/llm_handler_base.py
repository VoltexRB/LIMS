from abc import ABC, abstractmethod

class LLMHandlerBase (ABC):
    auth: dict
    host: str
    port: int
    default_parameters: dict
    model_name: str

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the service is connected

        :return: True if the service is connected
        """
        pass

    @abstractmethod
    def connect(self, data: dict) -> bool:
        """
        Connects to the external LLM Service

        :param data: Data to connect to the Service with, must contain at least "token" and "model"
        """
        pass

    @abstractmethod
    def send_prompt(self, prompt: str, rag: list[str] =None) -> dict:
        pass

    @abstractmethod
    def validate_model_name(self, model: str) -> bool:
        """
        Check if the model exists within the LLM Service

        :param model: Model name to check against
        :return: True if the model exists
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """
        Returns connection-data to be saved in the Settings-Object

        :return: Dict containing authentication-data
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the interface to use for dynamic binding (eg. "huggingface")
        """
        pass