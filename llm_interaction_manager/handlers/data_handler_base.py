from dataclasses import field
from abc import ABC, abstractmethod

#abstract
class DataHandlerBase(ABC):
    auth: dict
    host: str
    port: int
    default_parameters: dict

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the Data-Handler is connected to the external actor

        :return: True if the external actor is connected
        """
        pass

    @abstractmethod
    def connect(self, host: str, port: int, auth: dict = None) -> bool:
        """
        Connect to an external DataHander instance.

        :param host: Host-URL to connect to
        :param port: Port to connect to
        :param auth: Additional authentication, for example username or password
        :return: True if the service is successfully connected
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """
        Gets the current specific connection-info, so host, port, and auth for example

        :return: Dict-Object containing connection information
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the interface to use for dynamic binding (eg. "huggingface")
        """
        pass