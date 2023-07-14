"""
Utility scripts
"""

# External modules
import os
import openai
import tiktoken
import numpy as np
import pandas as pd
from time import time
from openai import ChatCompletion, Embedding
# Internal params
from econchatr.src.params import openai_key_name, model_choice_dict, embedding_model_name, embedding_dim_max
# Load encoding
enc = tiktoken.get_encoding('cl100k_base')


def set_openai_keys() -> None:
    """
    Will set the openai.api_key using the openai_key_name found in the params.py file
    """
    openai.api_key = os.getenv(openai_key_name)


def trim_double_quotes(x:pd.Series) -> pd.Series:
    """Convenience wrapper to remove starting/ending double quotes"""
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    z = x.str.replace('^\\"|\\"$','',regex=True)
    return z


def find_gpt_models(model_choice:str or None) -> list or None:
    """
    Checks to see if model_choice is a valid
     
    Args
    ====
    model_choice: str or None
        If model_choice is a string, will compare it to the accessible gpt models, if None, will query the gpt model list
    
    Returns
    =======
    None (assertion check only) if model_choice is a string, a list otherwise if model_choice is None
    """
    valid_models = pd.Series([m['id'] for m in openai.Engine.list()['data']])
    idx_gpt = valid_models.str.contains('gpt')
    gpt_models = "\n".join(valid_models[idx_gpt])
    print(f'Found the following {idx_gpt.sum()} GPT models:\n{gpt_models}')
    gpt_list = valid_models[idx_gpt].to_list()
    if model_choice is None:
        return gpt_list
    else:
        assert model_choice in gpt_list, f'Could not find {model_choice} in the GPT model list'


def n_tokens(txt:str) -> int:
    """
    Count the number of tokens in a string
    
    Args
    ====
    txt: str
        Some text to count the number of tokens for

    Returns
    =======
    An int with the number of tokens
    """
    return len(enc.encode(txt))


def truncate_tokens(txt:str, max_tokens:int=embedding_dim_max) -> str:
    """
    Truncates a txt to the first {max_tokens} length of tokens

    Args
    ====
    txt: str
        dd

    Returns
    =======
    """
    assert isinstance(txt, str), 'txt needs to be a string'
    split = enc.encode(txt)
    if len(split) > max_tokens:
        split = split[:max_tokens]
    return enc.decode(split)


def get_embeddings(txt:str or list, transpose:bool=False, model:str=embedding_model_name) -> np.ndarray:
    """
    Wrapper around the Embedding OpenAI API

    Args
    ====
    txt: str or list
        A string of list of chunks to be embedded
    transpose: bool
        Whether array should be tranposed so the columns are the number of the samples....
    model: str
        A valid model to use (defaults to ada-002)
    
    Returns
    =======
    An transpose((len(txt),1536)) numpy array of embedding values
    """
    # Input processesing
    assert isinstance(transpose, bool), 'transpose needs to be a bool'
    if not isinstance(txt, pd.Series):
        txt = pd.Series(txt)
    # Ensure the lengths work
    txt = txt.apply(truncate_tokens)
    # Run the API call
    response = Embedding.create(input=txt.to_list(), model=model)
    embeddings = np.vstack([r['embedding'] for r in response['data']])
    if transpose:
        return embeddings.T
    return embeddings


def complete_prompt(user:str, model:str or dict or None=None, system:str or None=None, verbose:bool=False, ret_time:bool=False, **kwargs) -> str:
    """
    Wrapper around the ChatCompletion function. Return a string (assumes that n=1)

    Args
    ====
    user: str
        The prompt that will be used by the user
    model: str or dict or None
        Give a valid model name (str) or a dict where the key is the model name, and the value is the token limit. If input is left blank, will default to the model_choice_dict dict in the params.py file. The model name will be based on the infimum of the token size.
    system: str or None (optional)
        Whether to give the chatbot a persona (defaults to helpful assistant)
    verbose: bool
        Whether the run time should be printed
    ret_time: bool
        Whether the a tuple of the text completion and runtime should be returned
    kwargs: **dict
        Other named arguments that will get passed into the function

    Returns
    =======
    A string which is the GPT output
    """
    # Input checks
    if 'n' in kwargs:
        assert kwargs['n'] == 1, 'complete_prompt, assumes that n==1'
    if system is None:
        system = 'You are a helpful assistant'
    assert isinstance(user, str), 'user arg needs to be a string'
    assert isinstance(system, str), 'system arg needs to be a string'
    # Add defaults to kwargs if not already present
    if 'temperature' not in kwargs:
        kwargs['temperature'] = 0.0
    if 'n' not in kwargs:
        kwargs['n'] = 1
    # Determine which model to use
    if model is None:
        model = model_choice_dict
    if isinstance(model, dict):
        # Calculate token lengths and pick model accordingly
        tot_tokens = n_tokens(user) + n_tokens(system)
        mx_tokens = pd.Series(list(model.values()))
        idx_tokens = mx_tokens >= tot_tokens
        assert idx_tokens.any(), f'Error! The total number of tokens in your user + system prompts {tot_tokens} exceeds limits {mx_tokens.max()}'
        model_choice = list(model.keys())[idx_tokens.idxmax()]
    else:
        find_gpt_models(model)  # Runs an assertion check
        model_choice = model
    # Construct the message list to match OpenAI API format
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content':user}]
    # Run the API call
    stime = time()
    res = ChatCompletion.create(model=model_choice, messages=messages, **kwargs)
    if verbose:
        dtime = time() - stime
        print(f'Took {dtime:0.0f} seconds to run the query')
    # Return the output
    txtout = res['choices'][0]['message']['content']
    if ret_time:
        return txtout, dtime
    return txtout
