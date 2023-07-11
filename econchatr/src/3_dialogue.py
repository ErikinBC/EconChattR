"""
Script which generates the simulated EconTalk transcript with Adam Smith.
"""

# External modules
import os
import pandas as pd
# Internal modules
from econchatr.src.params import path_dialogue
from econchatr.src.utils import set_openai_keys, n_tokens, complete_prompt

# Set up OpenAI connect
set_openai_keys()

# Prepare initial content
context = 'You are Russ Roberts, the host of the well-known podcast EconTalk. Today you have the once-in-a-lifetime opportunity to interview your intellectual hero who has returned from the dead for one day: the great 18th century Scottish economist Adam Smith, who is considered the founder of Economics and is most famous for his books the Wealth of Nations and the Theory of Moral Sentiments. Remember you wrote a book about Adam Smith called "How Adam Smith Can Change Your Life"'
print(f'Initial context has {n_tokens(context)} tokens')

# Load previous data
assert os.path.exists(path_dialogue), f'Could not find data: {path_dialogue}'
df_dialogue = pd.read_csv(path_dialogue)
txt_dialogue = df_dialogue['txt'].copy()


######################################
# --- (1) FIND ADAM SMITH QUOTES --- #

# Loop through any dialogue and find the sentences with Adam Smith references
idx_smith = txt_dialogue.str.contains('Smith',regex=False,na=False)
idx_smith = idx_smith & (df_dialogue['final_speaker'] == 'Russ')
txt_smith = txt_dialogue[idx_smith].reset_index(drop=True)
ntokens_smith = n_tokens(' '.join(txt_smith))
print(f'Found {idx_smith.sum()} Adam Smith dialogue lines that are a total of {ntokens_smith} tokens')

# Get GPT-4 to pull out the best sentences...
holder_smith_quotes = []
for i, quote in enumerate(txt_smith):
    print(f'Iteration {i+1} of {len(txt_smith)}')
    user = f"I'm building a chatbot that takes quotes from podcast episode transcripts and tries to guess what a podcast host would ask next. In the triple backquotes below you'll find a part of a transcript related to Adam Smith. Please extract the sentences or ideas, and format them so that they can be used as a reference later (e.g. 'Adam smith would have said this...', or 'My reading of Adam Smith is that...')```{quote}```\n\nReference material="
    #print(user)
    complete_prompt(user=user, verbose=True)

# for l in holder_smith_quotes.sample(20,random_state=1).to_list():
#     print(l);print('\n')

########################################
# --- (1) PREPARE INTERVIEW AGENDA --- #

# Find any Adam smith references
# Combine the content with the additional prompt for generating questions
prompt_agenda = f'To help prepare for the interview, below are the ten topics and/or questions we will have for Adam Smith. Remember, Adam Smith has been dead for hundreds of years, so we cannot ask him about current events or economic developments, nor can we use words of terms he would not have heard of in the 18th century. Instead we want to ask him questions about his theories and unresolved questoins Russ has\n\n1)'
print(prompt_agenda)


###################################
# --- (2) GENERATE TRANSCRIPT --- #



txt = "You've prepared interview notes for the converstation with Adam Smith that you will try to follow, but you will also let the conversation evolve naturally."