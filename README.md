# SIIM-FISABIO-RSNA-COVID-19-Detection
SIIM-FISABIO-RSNA COVID-19 Detection Kaggle Competition...

## Setup
No requirements txt is currently available for this project yet...

## Config
An example config is [here](config/example.yml) 

# Train
Simply run <code>python train.py --config [config name]</code>

## Motivation
I wanted to create a plug and play version of training. By utilising the config file we can keep adding parts (metrics, loss functions, checkpoints etc...) without removing old code.

## To Do
- Add more dataset variations
- Automatic Visualisation from Logs
- Inference program
- Add models to model zoo