"""
Utility scripts
"""

import os
import openai
import pandas as pd
from time import sleep
from datetime import datetime
local_tzname = datetime.now().astimezone().tzname()
from cleantext import clean
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def set_openai_keys() -> None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORG_ID")


def n_tokens(x:str) -> int:
    """Count the number of tokens in a string"""
    assert isinstance(x, str), 'Input must be a string'
    return len(tokenizer.encode(x))

def clean_text(txt:str) -> str:
    """Function to clean text"""
    res = clean(txt,
        fix_unicode=True,  to_ascii=True, lower=False,
        no_currency_symbols=False, no_punct=False,
        no_line_breaks=False, no_numbers=False, no_digits=False, 
        no_urls=True, no_emails=True, no_phone_numbers=True,
        replace_with_url="<URL>", replace_with_email="<EMAIL>", replace_with_phone_number="<PHONE>", replace_with_number="<NUMBER>", replace_with_digit="0", replace_with_currency_symbol="<CUR>",lang="en"
    )
    return res


def find_uploaded_data() -> dict:
    """Find the currently uploaded data files"""
    uploaded_files = openai.File.list()['data']
    uploaded_names = [file['filename'] for file in uploaded_files]
    uploaded_id = [file['id'] for file in uploaded_files]
    di_uploaded_data = dict(zip(uploaded_names, uploaded_id))
    return di_uploaded_data

def find_finetuned_models() -> dict:
    """Find the currently fine-tuned models"""
    finetuned_cloud = openai.FineTune.list()['data']
    # Only look at models that succeesed
    finetuned_cloud = [x for x in finetuned_cloud if x['status'] == 'succeeded']
    finetuned_model = [x['model'] for x in finetuned_cloud]
    finetuned_names = [x['fine_tuned_model'] for x in finetuned_cloud]
    di_finetuned_models = dict(zip(finetuned_model, finetuned_names))
    return di_finetuned_models


def wait_for_messages(openai_id:str, second_pause:int=30, terminal_message:str='Fine-tune succeeded') -> None:
    """
    Queries the openai.FineTune.list_events function to see which messages have been sent, and waits until the terminal message is sent
    """
    print(f'--- Waiting for messages from OpenAI for {openai_id} ---')
    keep_waiting = True
    # Loop until the terminal message is sent
    while keep_waiting:
        # Get the current time
        timerightnow = datetime.now().astimezone()
        # Get the messages
        df_messages = process_openai_messages(openai_id)
        # Print messages that have arrived since last update
        idx_print = (timerightnow >= df_messages['time']) & (df_messages['time'] + pd.DateOffset(seconds=second_pause) >= timerightnow)
        if len(idx_print) > 0:
            print('\n'.join(df_messages.loc[idx_print, 'message']))
        print(f'Time right now: {timerightnow}')
        # Check to see if the terminal message is in the messages
        keep_waiting = terminal_message not in df_messages['message'].values
        # Wait for a bit
        sleep(second_pause)


def process_openai_messages(openai_id:str) -> pd.DataFrame:
    """
    Process the messages from OpenAI
    """
    data = openai.FineTune.list_events(openai_id)['data']
    messages = pd.Series([x['message'] for x in data])
    date = pd.Series([x['created_at'] for x in data])
    date = pd.to_datetime(0, utc=True) + pd.TimedeltaIndex(date, unit='s')
    # Convert datetime from UTC to EST
    date = pd.Series(date).dt.tz_convert(local_tzname)
    # Convert to dataframe
    res = pd.DataFrame({'time':date,'message':messages})
    return res
        