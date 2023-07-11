"""
Script to process the transcript output from scrape.R
"""

# External modules
import os
import re
import numpy as np
import pandas as pd
# Internal modules
from econchatr.src.params import path_dialogue


###########################
# ---- (1) LOAD DATA ---- #

# Load raw_transcripts.txt line by line
path_transcripts = os.path.join('data','raw_transcripts.txt')
assert os.path.exists(path_transcripts), 'could not find raw_transcripts in the data folder'
with open(path_transcripts, 'r') as f:
    lines = pd.Series(f.readlines())
print(f'Found a total of {len(lines)} raw transcript lines')

# Find lines that start with "TimePodcast Episode"
start_lines = np.where(lines.str.contains('TimePodcast Episode'))[0]
n_start_lines = len(start_lines)

# Loop over start lines, and extract the text between each start line and the next start line
transcripts = []
for i in range(n_start_lines):
    if i == n_start_lines - 1:
        transcript = sum([],lines[start_lines[i]:].to_list())
    else:
        transcript = sum([],lines[start_lines[i]:start_lines[i+1]].to_list())
    transcripts.append(transcript[0])
# Convert to series
transcripts = pd.Series(transcripts)
# Throw error if transcripts does not start with "TimePodcast Episode"
assert transcripts.str.contains('TimePodcast Episode').all(), 'Expected to find TimePodcast at the start of all podcasts'
print(f'Found {len(transcripts)} episodes with time stamps')


############################
# ---- (2) CLEAN DATA ---- #

# Keep only transcripts that have a recording date
episodes = transcripts[transcripts.str.contains('Recording date')].reset_index(drop=True)
# Split on the first mention of recording date
rec_date = episodes.str.split(pat='Recording date',n=1,expand=True)[1]
# Remove the first instance of a colon and any whitespace that follows
rec_date = rec_date.str.replace('^\\:\\s','',regex=True)
# Keep only letters, numbers, and whitespace
rec_date = rec_date.str.replace('[^\\w\\s]',' ',regex=True)
# Replace multiple spaces with a single space
rec_date = rec_date.str.replace('\\s+',' ',regex=True)

# Extract the month with proceeds the first white space
tmp = rec_date.str.split(pat='\\s',n=1,expand=True)
rec_month = tmp[0].str.strip()
# Convert abbreviations to full month names for all months using a dictionary
month_dict = {'Jan':'January','Feb':'February','Mar':'March','Apr':'April','May':'May','Jun':'June','Jul':'July','Aug':'August','Sep':'September','Oct':'October','Nov':'November','Dec':'December'}
rec_month = rec_month.replace(month_dict)
# Replace Summer with July
rec_month = rec_month.str.replace('Summer','July')

# Clean up
rec_dyear = tmp[1].str.strip().str.replace(' century ', ' ')
# Replace a number with "th" or "rd" with the number
rec_dyear = rec_dyear.replace('^(\\d+)(st|nd|rd|th)\\s','\\1 ',regex=True)
# Remove the space if the day is two digits with a space between it
rec_dyear = rec_dyear.str.replace('^(\\d{1})\\s(\\d{1})\\s','\\1\\2 ',regex=True)
# Split tmp on the first character
rec_dyear = rec_dyear.str.split(pat='[a-zA-Z]',n=1,expand=True)[0]
# Split on spaces, with the first split being the day, and the second being the year
rec_dyear = rec_dyear.str.split(pat='\\s',n=2,expand=True)[[0,1]]
# Give dataframe column names
rec_dyear.columns = ['day','year']
# Empty strings are converted to NaN
rec_dyear = rec_dyear.replace('',np.nan)
# If the year column is missing, assign the value of the day, and replace the day with the value 1
rec_dyear = rec_dyear.assign(year=lambda x: np.where(x['year'].isnull(), x['day'], x['year']))
rec_dyear = rec_dyear.assign(day=lambda x: np.where(x['day']==x['year'],1,x['day']))
rec_dyear = rec_dyear.astype(str)

# Combine the month, day, and year into a datetime object
rec_dtime = pd.to_datetime(rec_month + ' ' + rec_dyear['day'] + ' ' + rec_dyear['year'],format='%B %d %Y')

# Check that rec_dtime is the same length as episodes
assert len(rec_dtime) == len(episodes), 'date length should make epsidoe length'
print(f'Found a total of {len(episodes)} episodes with recording dates')


#####################################
# ---- (3) CONVERT TRANSCRIPTS ---- #

# Find the years in the which all transcripts have Russ/Roberts broken down
n_russ = episodes.str.count('Russ:|Roberts:')
year_remove = rec_dtime[n_russ < 10].dt.year.max()
print(f'All transcripts have Russ/Roberts broken down before {year_remove}')
russ_episodes = episodes[rec_dtime.dt.year > year_remove]

# Print the episode max to min range using YYYY-MM-DD format
rec_dtime = rec_dtime[rec_dtime.dt.year > year_remove]
print(f'Episode date range: {rec_dtime.max().strftime("%Y-%m-%d")} to {rec_dtime.min().strftime("%Y-%m-%d")}')


# Enumerate over episode and determine speakers
pct_thresh = 0.1
holder = []
for i, s in enumerate(russ_episodes):
    if ((i+1) % 25) == 0:
        print(i+1)
    # Find all words before the colon
    speakers = re.findall(r'[^:]+', s, re.M)[1:-1]
    # Take the last two words
    speakers = pd.Series([speaker.split(' ')[-2:] for speaker in speakers])
    # Put them as  the two columns
    speakers = speakers.explode().rename_axis('idx').reset_index()
    speakers['cc'] = speakers.groupby('idx').cumcount()
    speakers = speakers.pivot(index='idx',columns='cc',values=0)
    # Transcript looks like: What do you think?Russ Roberts: .....
    speakers[0] = [q[-1] for q in speakers[0].str.split(pat='[^a-zA-Z]')]
    speakers[0] = speakers[0].str.replace('[^\w]','',regex=True).fillna('')
    speakers = speakers[0] + ' ' + speakers[1]
    speakers = speakers.astype(str).str.strip()
    # Speaker should not have a number, and should have at least one capital
    speakers = speakers[~speakers.str.contains('[0-9]')]
    speakers = speakers[speakers.str.contains('[A-Z]')].reset_index(drop=True)
    pct_speakers = speakers.value_counts(True)
    # Only keep speakers that are above the threshold
    final_speakers = pd.Series(pct_speakers[pct_speakers > pct_thresh].index)
    n_speakers = len(pct_speakers[pct_speakers > pct_thresh])
    # Check that the speaker + colon is in the transcript
    for final_speaker in final_speakers:
        n_lines = pd.Series(s).str.count(final_speaker + ':')[0]
        assert n_lines >= 7, f'Could not find 7 references for {final_speaker}'
    
    # Look over each of the final speakers and find the string location of the start of their speech
    # This is done by finding the first instance of the speaker + colon
    pos_speaker = []
    for final_speaker in final_speakers:
        pat = final_speaker+':'
        tmp = pd.DataFrame([(q.start(), q.end()) for q in re.finditer(pat, s, re.M)])
        tmp.columns = ['left','right']
        tmp['speaker'] = final_speaker
        pos_speaker.append(tmp)
    pos_speaker = pd.concat(pos_speaker).sort_values('left').reset_index(drop=True)
    # Convert the speaker column to an integer
    pos_speaker['code'] = pos_speaker['speaker'].astype('category').cat.codes
    pos_speaker['code'] = pos_speaker['code'].diff().fillna(1).astype(int)
    pos_speaker['code'] = np.abs(pos_speaker['code']).clip(0,1)
    pos_speaker['segment'] = pos_speaker['code'].cumsum()
    # start is the equivalent to the "right" column, and stop is the equivalent to the next "left" column
    pos_speaker['start'] = pos_speaker['right']
    pos_speaker['stop'] = pos_speaker['left'].shift(-1).fillna(len(s)).astype(int)
    # Add the text to each rows
    pos_speaker['txt'] = pos_speaker.apply(lambda x: s[x['start']:x['stop']],axis=1)
    # Groupby speaker and segment, and join the text
    pos_speaker = pos_speaker.groupby(['speaker','segment'])['txt'].apply(' '.join).reset_index()
    # Clean up the text removing excess white space
    pos_speaker['txt'] = pos_speaker['txt'].str.replace('\\s+',' ',regex=True).str.strip()
    # Add the episode number
    pos_speaker['episode'] = i
    
    # Append to holder
    holder.append(pos_speaker)
# Combine the holder
clean_speakers = pd.concat(holder)[['episode','segment','speaker','txt']]
clean_speakers = clean_speakers.sort_values(['episode','segment']).reset_index(drop=True)
print(f"There are a total of {clean_speakers['episode'].nunique()} episodes with Russ/Guest")


####################################
# ---- (4) CONVERT TO PROMPTS ---- #

# If the "speaker" column is not "Russ" or "Russ Roberts", then assign it to be "Guest"
clean_speakers['final_speaker'] = np.where(clean_speakers['speaker'].str.contains('^Russ'),'Russ','Guest')
# For each episode, final the Guest/Russ pairs the we can use the pivot table to create the prompt and completion columns
holder = []
for i, df in clean_speakers.groupby('episode'):
    # If Russ is not the first speaker, append a row to the top of the dataframe where the final_speaker is "Russ" and the txt is empty
    if df['final_speaker'].iloc[0] != 'Russ':
        df = pd.concat([pd.DataFrame({'episode':[i],'segment':[0],'speaker':['Russ'],'txt':[''],'final_speaker':['Russ']}),df])
    # Sort by segment and reset the index
    df = df.sort_values('segment').reset_index(drop=True)
    df['final_codes'] = df['final_speaker'].astype('category').cat.codes
    # If the final speaker does not alternate between Russ/Guest, then aggregate the rows of Guest
    check = df.drop(columns=['txt','episode']).assign(check=lambda x: x['final_speaker'].shift(1) != x['final_speaker'])['check'].all()
    if not check:
        # If the final_speaker value does not differ from the previous row, then aggregate the rows of Guest
        df = df.assign(segment=lambda x: x['final_codes'].diff().fillna(1).astype(int).abs().cumsum())
        # Check that each segment has at most one final_speaker value
        assert df.groupby('segment')['final_speaker'].nunique().max() == 1, f'Error in episode {i}'
        df = df.groupby(['episode','segment','final_speaker']).apply(lambda x: x['txt'].str.cat(sep=' ')).reset_index()
        df.rename(columns={0:'txt'},inplace=True)
    # Check that odd numbered rows have the final_speaker value of "Guest" and even numbered rows have the final_speaker value of "Russ"
    assert len(np.intersect1d(df['final_speaker'].iloc[::2].unique(),df['final_speaker'].iloc[1::2].unique())) == 0 
    # Check that Russ is the first speaker
    assert df['final_speaker'].iloc[0] == 'Russ', f'Error in episode {i}'
    # Append to holder
    holder.append(df)
# Combine the holder
df_russ_guest = pd.concat(holder).drop(columns=['final_codes','speaker']).reset_index(drop=True)
df_russ_guest['txt'] = df_russ_guest['txt'].str.replace('([0-9]{1}\\:)?[0-9]{1,2}\\:[0-9]{1,2}','',regex=True)
path_dialogue = os.path.join('data', 'russ_guest.csv')
df_russ_guest.to_csv(path_dialogue, index=False)

# # Grouping by each episode, the "prompt" is the previous row of txt, and the "completion" is the current row of txt
# df_russ_guest['prompt'] = df_russ_guest.groupby('episode')['txt'].shift(1).fillna('')
# df_russ_guest['completion'] = df_russ_guest['txt']
# prompt_completion = df_russ_guest.drop(columns='txt').query('final_speaker=="Russ"')
# # Groupby episode and replace segment with cumcount
# prompt_completion['segment'] = prompt_completion.groupby('episode')['segment'].cumcount()+1
# prompt_completion.reset_index(drop=True, inplace=True)

# # Save prompt_completion as JSONL file for the prompt and completion columns
# prompt_completion[['prompt','completion']].to_json(os.path.join('data','prompt_completion.jsonl'),orient='records',lines=True)

print('~~~ End of 2_process_transcripts.py ~~~')