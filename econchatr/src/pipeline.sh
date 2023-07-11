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
Rscript econchatr/src/1_scrape.R
#   output: data/raw_transcripts.txt

# Clean up raw_transcripts for use in model
echo "--- (2) Running process_transcripts.py --- "
python3 -m econchatr.src.2_process_transcripts
#   output: data/russ_guest.csv

# Generate the podcast transcript
echo "--- (3) Running 3_dialogue.py ---"
#   output: 


echo "~~~ End of pipeline.sh ~~~"