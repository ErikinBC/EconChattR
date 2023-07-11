"""
Stores the openai parameters
"""

# Data file name
path_dialogue = 'data/russ_guest.csv'

# Define the name of variable name of the openAI key and group
openai_key_name = 'OPENAI_PLAYGROUND_KEY'
opeani_org_name = 'OPENAI_PLAYGROUND_ORG'

# Set the model we will use
model_choice_dict = {'gpt-4':8192, 'gpt-3.5-turbo-16k':16000}

# Set up the baseline parameter dictionary (model and prompt will need to swapped in)
di_completion_params = {
  "max_tokens": 250,
  "temperature": 0.6,
  "n": 1,
  "echo": True,
  "stream": False,
}

# Model list to test
models = {'ada':'text-ada-001',
          'babbage':'text-babbage-001',
          'curie':'text-curie-001', 
          'davinci':'text-davinci-003'}


# Prompt list to test (informed by the EconTalk drinking game: 
# https://www.etsy.com/ca/listing/696468353/econtalk-drinking-game-limited-edition)
prompts = [
  'Does Russ Roberts mention the "Bootlegger and Baptist" theory often?',
  'As a proponent of free markets, is Russ Roberts attracted to the "Bootlegger and Baptist" theory?',
  'How skeptical is Russ Roberts of empirical research published in academic journals?',
  'Does Russ Roberts believe that "data can speak for itself" as a general principle?',
  'Finish the following sentence: "Man desires not only to be loved"',
  'Finish the following sentence: "The curious task of economics"',
  'Why is "skin in the game" a deep insight?', 
  'In what way does understanding the challenge around "creating a prairie" align with F A Hayek\'s famous maxim about the curious task of economics?',
  'Why is it harder to "create a prairie" then it seems?',
  'As a free market economist, why is Russ Roberts drawn to the idea of the "Chesterton Fence"?',
  'Finish the following sentence: "Macroeconomists have a sense of humor because"',
  'Who are some guests from EconTalk from the year 2014?'
  ]


