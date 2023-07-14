"""
Script which generates the simulated EconTalk transcript with Adam Smith.
"""

# External modules
import os
from time import time
import pandas as pd
# Internal modules
from econchatr.src.params import path_dialogue, path_smith_quotes
from econchatr.src.utils import set_openai_keys, n_tokens, complete_prompt, get_embeddings, trim_double_quotes

# Set up OpenAI connect
set_openai_keys()

# Load previous data
assert os.path.exists(path_dialogue), f'Could not find data: {path_dialogue}'
df_dialogue = pd.read_csv(path_dialogue)
txt_dialogue = df_dialogue['txt'].copy()


######################################
# --- (1) FIND ADAM SMITH QUOTES --- #

# (i) Loop through any dialogue and find the sentences with Adam Smith references
idx_smith = txt_dialogue.str.contains('Smith',regex=False,na=False)
idx_smith = idx_smith & (df_dialogue['final_speaker'] == 'Russ')
txt_smith = txt_dialogue[idx_smith].reset_index(drop=True)
ntokens_smith = n_tokens(' '.join(txt_smith))
print(f'Found {idx_smith.sum()} Adam Smith dialogue lines that are a total of {ntokens_smith} tokens')

# (ii) Ask GPT to summarize each of the quotes
holder_smith_summary = []
mu_dtime = 0
nquotes = len(txt_smith)
for i, quote in enumerate(txt_smith):
    print(f'Iteration {i+1} of {len(txt_smith)}')
    user = f"""
    You are a podcast host who is preparing for an upcoming interview with Adam Smith, the 18th-century Scottish economist. You've brought up many of Smith's ideas in your previous episodes, and want to prepare a list of questions and quotes to better understand his ideas. The triple backquotes below contain the verbatim transcript from one of your episodes. Please extract the main idea or quotes related to Adam Smith from this text, if there is any, in a single sentence in the first person that can be used later in a bullet point reference list.  Remember, this extraction will be used as the basis for asking Adam Smith questions or responding to his answers.
    \n
    ```{quote}```
    \n
    Main idea/quote:"""
    outtxt, dtime = complete_prompt(user=user, verbose=True, ret_time=True)
    mu_dtime = i/(i+1)*mu_dtime + dtime/(i+1)
    nleft = nquotes - (i+1)
    seta = nleft * mu_dtime
    print(f'ETA = {seta:0.0f} seconds ({seta/60:0.1f} minutes)')
    holder_smith_summary.append(outtxt)
# Combine into single DF and save
df_smith_quotes = pd.DataFrame({'summary':holder_smith_summary, 'raw':txt_smith})
df_smith_quotes['summary'] = trim_double_quotes(df_smith_quotes['summary'])
df_smith_quotes['ntok_summary'] = df_smith_quotes['summary'].apply(n_tokens)
df_smith_quotes['ntok_raw'] = df_smith_quotes['raw'].apply(n_tokens)
sumstats = df_smith_quotes['ntok_summary'].describe().astype(int)
print(f"Total of number summarized tokens {sumstats['count']*sumstats['mean']} (average={sumstats['mean']}, min={sumstats['min']}, max={sumstats['max']})")


############################
# --- (2) EMBED QUOTES --- #

# (i) Run on summary and raw notes
emb_smith_summary = get_embeddings(df_smith_quotes['summary'])
emb_smith_raw = get_embeddings(df_smith_quotes['raw'])
assert emb_smith_raw.shape == emb_smith_raw.shape, 'How are these embeddings not the same shape?!'

# (ii) Find the "non-relevant" content
txt_nonrelevant = 'This transcript does not contain any references to Adam Smith or his ideas.'
emb_nonrelevant = get_embeddings(txt_nonrelevant)

# (iii) Flag anything that's above 85%
df_smith_quotes['cossim'] = emb_nonrelevant.dot(emb_smith_summary.T).flatten()
df_smith_quotes['relevant'] = df_smith_quotes['cossim'] < 0.85
df_smith_quotes.to_csv(path_smith_quotes, index=False)


########################################
# --- (3) PREPARE INTERVIEW AGENDA --- #

# Split the summary quotes into three groupings, and summarize five points from each
nbin = 3
df_smith_quotes = df_smith_quotes.assign(tot_tokens=lambda x: (x['relevant']*x['ntok_summary']).cumsum())
df_smith_quotes['bins'] = pd.cut(df_smith_quotes['tot_tokens'], nbin).cat.codes

holder_interview_agenda = []
for bin in df_smith_quotes['bins'].unique():
    print(f'Bin {bin+1} of {nbin}')
    txt_summary_bin = df_smith_quotes.query('bins==@bin & relevant')['summary']
    txt_summary_bin = "\n".join('```' + txt_summary_bin + '```')
    user = f"""You are a podcast host preparing for an interview with Adam Smith. You've compiled a list summarizing previous things you've said or quoted about Adam Smith in triple back quotes and you want to now consolidate this into five question topics for your upcoming interview. Each of these five interview notes should 2-3 sentences long, written in first person, and capture a key idea or quote from Adam Smith that you will base your interview questions on. 
    \n
    {txt_summary_bin}
    """
    outtxt, dtime = complete_prompt(user=user, verbose=True, ret_time=True)
    print(f'Time to query: {dtime:0.0f} seconds')
    holder_interview_agenda.append(outtxt)
# Combine and renumber
interview_agenda = pd.concat([pd.Series(h.split('\n\n')) for h in holder_interview_agenda])
interview_agenda = interview_agenda.str.split('\\.\\s',n=1,expand=True)[1]
interview_agenda = trim_double_quotes(interview_agenda).reset_index(drop=True)

# Futher consolidation???
concat_interview_agenda = '\n'.join(interview_agenda)
system_concat = 'You are are podcast host preparing to interview Adam Smith, the long-dead Scottish economist and philosopher who is considered the founder of economics.'
user_concat = f"""I've prepared a list of interview topics for my upcoming interview with Adam Smith. Please rank-order the following list from best to worst in terms of how well we'd expect a historical figure to be able to answer them. You only need to list the numbers in a comma-separated fashion.
\n
{concat_interview_agenda}
"""
question_rank = complete_prompt(user=user_concat, system=system_concat)
idx_rank = pd.Series(question_rank.split(', ')[:10]).astype(int)-1
# Subset
interview_agenda = interview_agenda[idx_rank].reset_index(drop=True)
interview_agenda = (interview_agenda.index+1).astype(str) + '. ' + interview_agenda
print('\n'.join(interview_agenda))


###################################
# --- (4) GENERATE TRANSCRIPT --- #

# First sentence will be pre-scripted for Russ.
russ_system = """You are Russ Roberts, the host of the well-known podcast EconTalk. Today you have the once-in-a-lifetime opportunity to interview your intellectual hero who has returned from the dead for one day: the great 18th century Scottish economist Adam Smith, who is considered the founder of Economics and is most famous for his books the Wealth of Nations and the Theory of Moral Sentiments. Remember you wrote a book about Adam Smith called "How Adam Smith Can Change Your Life"."""
print(russ_system)

russ_start = "[Russ Roberts] Today is July 13, 2023 and my guest is Adam Smith, Professor of Political Economy. As many of you know, Mr. Smith has been dead for almost 200 years, but has generously resurrected himself for the purposes of our conversation today. Mr. Smith, welcome to EconTalk."

smith_system = 'You are Adam Smith, the renowned 18th-century economist and philosopher who was a pioneer in the thinking of political economy and a key figure during the Scottish Enlightenment. You speak as an 18th-century Scottish/English gentleman would by using elevated language and sentence lengths bordering on prolixity.'

editor_user = 'Are there any terms in your question that might be confusing to Mr. Smith? If so, re-write the question that preserves as much of the original text as possible but that addresses any confusing terms.'

# How much token context do we want?
mx_context = 5950

# We'll stop at around 70000 characters
nchar_stop = 70000
nchar_iter = 0 
i = 0 
smith_out = ''
holder_convo = []
stime = time()
while nchar_iter < nchar_stop:
    i += 1
    if i == 1:
        # Russ initializing with the usual EconTalk intro
        russ_out = russ_start
    else:
        russ_out = complete_prompt(user=smith_out, system=russ_system, top_p=0)
    smith_out = complete_prompt(user=russ_out, system=smith_system, top_p=0)
    if '[Adam Smith]' not in smith_out:
        smith_out = '[Adam Smith] ' + smith_out
    if smith_out.split('.')[-1] != '':
        print('Woah! We might have hit the token limit!')
        break
    # Store the dialogue pair
    di_out = {'russ':russ_out, 'smith':smith_out}
    holder_convo.append(di_out)
    
    # Estimate the remaining time
    dtime = time() - stime
    nchar_russ = len(russ_out)
    nchar_smith = len(smith_out)
    nchar_iter += nchar_russ + nchar_smith
    nchar_per_sec = nchar_iter / dtime
    seta = (nchar_stop - nchar_iter) / nchar_per_sec
    print(f'--- Dialogue pair {i} ---\nETA={seta:0.0f} seconds ({seta/60:0.1f} minutes)\nRuss={nchar_russ}, Smith={nchar_smith}')

    # Update the smith_out for Russ at the next iteration
    first_pair = f"{holder_convo[0]['russ']}\n\n{holder_convo[0]['smith']}"
    tot_tokens_for_russ = n_tokens(smith_out) + n_tokens(first_pair)
    tokens_left_for_russ = mx_context - tot_tokens_for_russ
    assert tokens_left_for_russ > 0, 'Woops, we dont have enough tokens!'
    

    other_pairs = ''
    smith_out = first_pair + other_pairs
    # Russ' token count is his system prompt plus the previous dialogue with smith
    

    break



# Combine the content with the additional prompt for generating questions
prompt_agenda = f'To help prepare for the interview, below are the ten topics and/or questions we will have for Adam Smith. Remember, Adam Smith has been dead for hundreds of years, so we cannot ask him about current events or economic developments, nor can we use words of terms he would not have heard of in the 18th century. Instead we want to ask him questions about his theories and unresolved questoins Russ has\n\n1)'
print(prompt_agenda)

