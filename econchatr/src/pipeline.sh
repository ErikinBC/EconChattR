#!/bin/bash

# --- Pipeline to create EconChatR --- #

# Hard-coded parameters for model training
n_epochs=4  # Number of epochs to train the model
max_cost=10  # This is in USD


# (0) Set up conda environment if it doesn't already exist
# To build from scratch run: conda env create -f env.yml first
conda activate econtalk

# Get the EconTalk.xml file and transcripts
echo "--- (1) Running scrape.R --- "
Rscript 1_scrape.R
#   output: data/raw_transcripts.txt

# Clean up raw_transcripts to prompt/completion JSONL
echo "--- (2) Running process_transcripts.py --- "
python3 2_process_transcripts.py
#   output: data/prompt_completion.jsonl

# Prepare the data for training
echo "--- (3) Running prepare_training.py --- "
python3 3_prepare_training.py --n_epochs $n_epochs --max_cost $max_cost
#   output: data/training_data_all.jsonl
#           data/training_data_{model}.jsonl

# Use the OpenAI tools to train the model
echo "--- (4) Creating fine-tuned model with openai  --- "
python3 4_tune_models.py --n_epochs $n_epochs
#   output: see https://beta.openai.com/ (FINE-TUNED MODELS)

# Comparing trained model results to existing
echo "--- (5) Run prompt_baseline  --- "
python3 5_prompt_baseline.py


echo "~~~ End of pipeline.sh ~~~"