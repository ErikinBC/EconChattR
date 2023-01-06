"""
Take output from process_transcripts and select "high quality" training samples and estimate cost
"""

# Set up argparse to include calculates for n_epochs, and max_cost
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs to run')
parser.add_argument('--max_cost', type=float, default=10, help='Maximum cost per model ($USD)')
args = parser.parse_args()
n_epochs = args.n_epochs
max_cost = args.max_cost

import os
import numpy as np
import pandas as pd
from scipy import stats
# Internal imports
from cost import calculate_epoch_cost
from utils import n_tokens, clean_text


#########################
# --- (1) LOAD DATA --- #

# Load the prompt_completion JSONL data
prompt_completion = pd.read_json(os.path.join('data','prompt_completion.jsonl'), lines=True)
    
# Convert to a Pandas DataFrame
df = pd.DataFrame(prompt_completion).rename_axis('id').reset_index()
df = df.melt('id', var_name='pair', value_name='txt')
df = df.sort_values(['id','pair'],ascending=[True,False]).reset_index(drop=True)


############################
# --- (2) CLEAN TOKENS --- #

# Remove repeated words
df['txt'] = df['txt'].apply(lambda x: ' '.join([t for i,t in enumerate(x.split()) if i==0 or t != x.split()[i-1]]))

# If a character follows a comma, question mark, or period without a space, add a space
punct_with_spaces = '!,.?'
df['txt'] = df['txt'].str.replace(f'([{punct_with_spaces}])', r'\1 ', regex=True)

# Anytime there is a punction followed by a space, followed by a ', remove the '
df['txt'] = df['txt'].str.replace(f'([{punct_with_spaces}])\s\'', r'\1 ', regex=True)

# Remove the [? ] from the text
df['txt'] = df['txt'].str.replace('\\[\\?\\s\\]', '',regex=True)

# Remove extra spaces
df['txt'] = df['txt'].str.replace('\s{2,}', ' ', regex=True).str.strip()

# Use the clean_text function to help with ascii issues
df['txt'] = df['txt'].apply(clean_text)

# Get approximate number of tokens in each
df['grps'] = pd.cut(df['txt'].str.len() / 4, [0, 10, 60, 200, np.inf], right=False, labels=['<10', '10-60', '60-200', '>200'])

# # Sample the data (uncomment for exploratory work)
# n_sample = 5
# for i, r in df.groupby(['pair','grps']).sample(n_sample,random_state=n_sample).iterrows():
#     print(f"row={i}, pair={r['pair']}, grps={r['grps']}, txt=\n{r['txt']}")
#     print('------')
#     print('\n')
#     input('Press Enter to continue...')

# Look at the pairs of prompts and completions
df.pivot('id','pair','grps').groupby(['completion','prompt']).size().unstack()


#####################################
# --- (3) CALCULATE TOKEN COUNT --- #

# Calculate the number of tokens found in the completion column using GPT2TokenizerFast
df['ntokens'] = df['txt'].apply(n_tokens)
# Calculate the number of words in the txt column
df['nwords'] = df['txt'].apply(lambda x: len(x.split()))
# Calculate the number of characters in the txt column
df['nchars'] = df['txt'].apply(lambda x: len(x))
# Print the average and total number of tokens, words, and characters
print(df.agg({'ntokens':['mean','sum'],'nwords':['mean','sum'],'nchars':['mean','sum']}).astype(int))

# Run a linear regression between number of words/characters and number of tokens
# Use scipy to do this
slope_tokens = stats.linregress(x=df['nwords'], y=df['ntokens'])[0]
print(f'For every one word, there is an average of {slope_tokens:.2f} tokens')
inv_slope_nchars = stats.linregress(x=df['ntokens'], y=df['nchars'])[0]
print(f'For every one token, there is an average of {inv_slope_nchars:.2f} characters')

# For the fine tuning dataset, we will ignore Russ' answers that are 20 tokens or fewer
# We will also strip out prompt/completions that exceed 2000 tokens, since there is 2048 limit
id_russ_min = df.query('pair=="completion" and ntokens>20')['id']
id_guest_min = df.query('pair=="prompt" and ntokens>60')['id']
id_token_max = df.pivot('id','pair','ntokens').sum(1).sort_values().reset_index().rename(columns={0:'ntokens'}).query('ntokens<=2000')['id']
id_keep = np.intersect1d(np.intersect1d(id_russ_min, id_guest_min), id_token_max)
training_data = df[df['id'].isin(id_keep)].reset_index(drop=True)
# Remove the duplicates (if any)
id_drop = training_data.pivot('id','pair','txt').duplicated().reset_index().rename(columns={0:'duplicates'}).query('duplicates')['id']
training_data = training_data[~training_data['id'].isin(id_drop)].reset_index(drop=True)
n_pairs_training = training_data['id'].nunique()
print(f"A total of {n_pairs_training} prompts/completions remain")
# Set pair to categorical to ensure correct order
training_data['pair'] = pd.Categorical(training_data['pair'], categories=['prompt','completion'])


##############################
# --- (4) SAVE FULL DATA --- #

idx_prompt = training_data['pair']=='prompt'
idx_completion = training_data['pair']=='completion'
# Add " Russ Roberts responds:" to the end of the prompt rows
training_data.loc[idx_prompt ,'txt'] = training_data.loc[idx_prompt ,'txt'] + ' Russ Roberts responds:'
# Add a space to the beginning of the completion rows
training_data.loc[idx_completion ,'txt'] = ' ' + training_data.loc[idx_completion ,'txt']
# Add a suffix ending `\n` to all completions
training_data.loc[idx_completion ,'txt'] = training_data.loc[idx_completion ,'txt'] + '\n'

# Adjust the token count from before (not we've added four tokens to the prompt, and 1 token to the completion)
training_data = training_data.assign(ntokens=lambda x: np.where(x['pair']=='prompt', x['ntokens']+4, x['ntokens']+1))
assert training_data.sample(100,random_state=1).assign(check=lambda x: x['txt'].apply(n_tokens,1) == x['ntokens'])['check'].all(), 'Token count does not match'
# Calculate the total number of tokens
n_tokens_training = training_data['ntokens'].sum()
# Calculate the total number of tokens associated with each id
n_tokens_id = training_data.pivot('id','pair','ntokens').sum(1).reset_index().rename(columns={0:'ntokens'})

# Save to a JSONL file by pivoting on the id, pair, and completion columns
training_data_wide = training_data.pivot(index='id', columns='pair', values='txt')
training_data_wide.to_json(os.path.join('data','training_data_all.jsonl'),orient='records',lines=True)


##################################
# --- (4) CREATE DATA SUBSET --- #

# Calculate the (approximate) model specific cost for all epochs
dat_cost = calculate_epoch_cost(None, n_tokens_training)
# If data reduction is >1, leave as is
data_reduction = (max_cost / (dat_cost * n_epochs)).clip(upper=1)
data_reduction = data_reduction.reset_index().rename(columns={'Training':'pct'})
# Calculate the approximate number of samples, and exact number of tokens
data_reduction = data_reduction.assign(n_pairs=lambda x: np.floor(x['pct']*n_pairs_training).astype(int))
data_reduction = data_reduction.assign(token_limit=lambda x: np.floor(n_tokens_training * x['pct']).astype(int))

# Determine the "high" quality samples that are non-negotiable
russ_txt = training_data.query('pair=="completion"')['txt']
regex_russ = 'bootlegger|baptist|empirical|skeptical|prairie|deep|Chesterton\\sfence|decimal\\spoint|macroeconomist|macroeconomic|regression'
idx_data_speak = russ_txt.str.contains('speak',regex=True) & russ_txt.str.contains('data',regex=True)
idx_data_speak = idx_data_speak[idx_data_speak==True].index.values
# Loop over each regular expression and find the associated rows, calculating the total number of tokens (note that if another category shares it, we will set them to zero, since they can share the same tokens)
di_regex_idx = {'data_speak':idx_data_speak}
for pat in regex_russ.split('|'):
    idx_pat = russ_txt.str.contains(pat,case=False,regex=True)
    idx_pat = idx_pat[idx_pat==True].index.values
    di_regex_idx[pat] = idx_pat
# Create a dataframe with the indices and the number of tokens
df_regex_idx = pd.DataFrame.from_dict(di_regex_idx,orient='index').T.melt(var_name='regex',value_name='idx').dropna().set_index('regex').astype(int)
df_regex_idx['id'] = training_data.loc[df_regex_idx['idx'],'id'].values
# Sum number of tokens for both prompt and completion and sort within regex
df_regex_idx = df_regex_idx.reset_index().merge(n_tokens_id).sort_values(['regex','ntokens']).reset_index(drop=True)
# Add a group-specific counter within regex
df_regex_idx = df_regex_idx.assign(number=lambda x: x.groupby('regex').cumcount()+1)
# For each number, get the unique id's
id_order_keep = df_regex_idx.groupby(['number'])['id'].unique().reset_index().explode('id').set_index('number')
# Remove duplicates from lower numbers
id_order_keep = id_order_keep[~id_order_keep.duplicated()]
# Add on the number of tokens
id_order_keep = id_order_keep.reset_index().merge(n_tokens_id)
# Get the ID's that are not part of the regex list
other_ids = np.setdiff1d(training_data['id'].unique(), id_order_keep['id'])
# Add them onto to DataFrame...
tmp = pd.DataFrame({'number':range(len(other_ids)),'id':pd.Series(other_ids).sample(frac=1,random_state=1,replace=False)}).merge(n_tokens_id)
tmp = tmp.assign(number=lambda x: x['number']+id_order_keep['number'].max()+1)
id_order_keep = pd.concat([id_order_keep,tmp]).reset_index(drop=True)
# Get the cumulative sum of the tokens
id_order_keep['ntokens'] = id_order_keep['ntokens'].cumsum()

# Loop through each of the data_reduction rows, and determine which samples to keep
for i, row in data_reduction.iterrows():
    token_limit, expected_pairs, model = row['token_limit'], row['n_pairs'], row['Model']
    # Find the ids that are less than the token limit
    ids_limit = id_order_keep.query('ntokens<=@token_limit')['id']
    # Print a comparison to the expected
    print(f'Number of pairs supported: {len(ids_limit)} (expected: {expected_pairs}) for {model}')
    # Save a data version
    training_data_wide.loc[ids_limit].to_json(os.path.join('data',f'training_data_{model.lower()}.jsonl'),orient='records',lines=True)


print('~~~ End of 3_prepare_training.py ~~~')