# CRISP: Curriculum based Sequential neural decoders for Polar code family

This repository contains code used to run experiments in the above paper. 

## For running the CRISP model 

The bash files `run_crisp.sh` trains using the best curricula and GRUs. By default we use the hyperparameters that gave us decent performance for a reasonable training time. They may be changed inside the file. 

## For running alternate models 

The bash files `run_alt.sh` trains using the best curricula and CNNs. Other models can be trained by changing the `--model` option. By default we use the hyperparameters that gave us decent performance for a reasonable training time. They may be changed inside the file.