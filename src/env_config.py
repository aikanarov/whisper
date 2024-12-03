"""
Module for managing environment configurations.
"""

from pydantic_settings import BaseSettings
from pydantic.types import SecretStr


class Settings(BaseSettings):
    """
    Class for managing settings.
    """

    # API keys and endpoints
    openai_api_key: SecretStr = "xxx"

    class Config:
        # File path for environment file
        env_file = '.env'
        # Directory path for secrets
        #secrets_dir = 'run/secrets'


# Creating an instance of Settings
env = Settings()