"""
Script that stores the different cost structures

See: https://openai.com/api/pricing/

As of Jan 4, 2023
"""

import os
import numpy as np
import pandas as pd
from utils import n_tokens

# Inference on existing LLMs
data = {'Model': ['Ada', 'Babbage', 'Curie', 'Davinci'],
        'Usage': ['$0.0004 / 1K tokens', '$0.0005 / 1K tokens', '$0.0020 / 1K tokens', '$0.0200 / 1K tokens']}
existing = pd.DataFrame(data).set_index('Model')
# There are per 1000 tokens
existing = existing.apply(lambda x: x.str.replace('[^0-9.]','',regex=True).astype(float)) / 1000

# Custom model
data = {'Model': ['Ada', 'Babbage', 'Curie', 'Davinci'],
        'Training': ['$0.0004 / 1K tokens', '$0.0006 / 1K tokens', '$0.0030 / 1K tokens', '$0.0300 / 1K tokens'],
        'Usage': ['$0.0016 / 1K tokens', '$0.0024 / 1K tokens', '$0.0120 / 1K tokens', '$0.1200 / 1K tokens']}
custom = pd.DataFrame(data).set_index('Model')
# There are per 1000 tokens
custom = custom.apply(lambda x: x.str.replace('[^0-9.]','',regex=True).astype(float)) / 1000


# Calculate the epoch cost for jsonl file
def calculate_epoch_cost(file:str or None, total_tokens:int or None=None) -> float:
        """
        Calculate the model-specific cost of training on a jsonl file

        Inputs
        ------
        file: str               Path to the jsonl file
        total_tokens: int       Number of tokens in the file (default: None)
        """
        if total_tokens is not None:
                assert isinstance(total_tokens, int) or isinstance(total_tokens, np.int64), 'total_tokens must be an integer'
        else:
                # Check that the file exists
                assert os.path.exists(file), f'File {file} does not exist'
                # Load the file
                df = pd.read_json(file, lines=True)
                # Calculate the number of tokens
                total_tokens = sum([n_tokens(s) for s in df.melt()['value']])
        # Calculate the cost
        cost = (custom * total_tokens)['Training'].round(2)
        print(f'Will cost the following per model {cost}')
        return cost
