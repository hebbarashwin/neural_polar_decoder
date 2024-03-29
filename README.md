# CRISP: Curriculum based Sequential neural decoders for Polar code family

This repository contains code used to run experiments in the above paper. 

## For running the CRISP model 

The bash files `run_crisp.sh` trains using the best curricula and GRUs. By default we use the hyperparameters that gave us decent performance for a reasonable training time. They may be changed inside the file. 

## For running alternate models 

The bash files `run_alt.sh` trains using the best curricula and CNNs. Other models can be trained by changing the `--model` option (can choose between `conv`(CNNs), `gpt`(GPT), `encoder`(BERT)). By default we use the hyperparameters that gave us decent performance for a reasonable training time. They may be changed inside the file. The curriculum can be changed with `--curriculum` option (choose between `l2r`, `r2l`, `c2n` and `n2c`).

## Python requirements

The required Python packages can be installed by using:
```
pip install -r requirements.txt
```
