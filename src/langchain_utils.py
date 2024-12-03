import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI

from src.env_config import env

# ---------------------
# Models
# ---------------------

def get_llm_openai(model_name: str, temperature: float = 0.7, max_tokens: int = 256, request_timeout: int = 60) -> ChatOpenAI:
    """
    This function creates and returns a language model instance based on the specified model name.

    Args:
        model_name (str): The name of the language model. Currently supports 'gpt-4', 'gpt-3.5-turbo' and 'text-davinci-003'.
        temperature (float, optional): The temperature for the language model. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens for the language model. Defaults to 256.
        request_timeout (int, optional): The request timeout for the language model. Defaults to 60.

    Returns:
        Union[ChatOpenAI, OpenAI]: An instance of either ChatOpenAI or OpenAI based on the model name.

    Raises:
        ValueError: If the model name is not supported.
    """

    return ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        max_tokens=max_tokens,
        openai_api_key=env.openai_api_key.get_secret_value()
    )

def get_llm_azure(model_name: str, temperature: float = 0.7, max_tokens: int = 256, request_timeout: int = 60) -> AzureChatOpenAI:
    """
    This function creates and returns a language model instance based on the specified model name.

    Args:
        model_name (str): The name of the language model.
        temperature (float, optional): The temperature for the language model. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens for the language model. Defaults to 256.
        request_timeout (int, optional): The request timeout for the language model. Defaults to 60.

    Returns:
        AzureChatOpenAI: An instance of AzureChatOpenAI based on the model name.

    Raises:
        ValueError: If the model name is not supported.
    """

    api_version = "2023-12-01-preview"

    model_classes = {
        "gpt-4": ['gpt-4'],
        "gpt-4-1106-preview": ['gpt-4'],
        "gpt-3.5-turbo": ['gpt-35-turbo'],
        "gpt-3.5-turbo-1106": ['gpt-35-turbo'],
    }

    if model_name not in model_classes:
        raise ValueError(
            f"Unsupported model name: {model_name}. Supported model names are: {list(model_classes.keys())}")

    llm =  AzureChatOpenAI(
        azure_deployment=model_classes[model_name][0],
        api_version=api_version,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_type=env.azure_openai_api_type.get_secret_value(),
        azure_endpoint=env.azure_openai_api_base.get_secret_value(),
        api_key=env.azure_openai_api_key.get_secret_value()
    )

    return llm

def get_llm_anthropic(model_name: str, temperature: float = 0.7, max_tokens: int = 256, request_timeout: int = 60) -> ChatAnthropic:
    """
    This function creates and returns a language model instance based on the specified model name.

    Args:
        model_name (str): in model names
        temperature (float, optional): The temperature for the language model. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens for the language model. Defaults to 256.

    """
    models = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    if model_name not in models:
        model_name = 'claude-3-opus-20240229'
        print(f'--- Model not found defaulting to {model_name}')

    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        anthropic_api_key=env.anthropic_api_key.get_secret_value(),
        max_tokens=max_tokens,
    )

# ---------------------
# Embeddings
# ---------------------

def get_embedder() -> OpenAIEmbeddings:
    """Get the OpenAIEmbeddings instance."""

    embeddings = OpenAIEmbeddings(
        openai_api_key=env.openai_api_key.get_secret_value(),
        model="text-embedding-ada-002"
    )

    return embeddings

