# THVAE

## Introduction

HEAD
We developed THVAE, an **unsupervised** timline summarization model based on hierarchical VAE 
=======

## Train
The main model is in [file](https://github.com/Maria-Liakata-NLP-Group/THVAE-summary/blob/main/copycat/modelling/thvae.py), it shows the process of how to use hierarchical VAE to get the latent code of each segment of a timeline.
[file](https://github.com/Maria-Liakata-NLP-Group/THVAE-summary/blob/main/copycat/modelling/interfaces/ithvae.py) has the mehod of how to construct the summary representation using key phrases, the method is 'predict'.
If you want to train with your own dataset, run 
```
python thvae/scripts/run_workflow.py
```
## Data

We experimented on talk-life datasets 

### Input Data Format

The expected format of input is 

group_id | review_text | category 
--- | --- | --- 
136861_255 | im worthless im literally a fucking failure . let me die | post 


## Key phrases

The method of getting key phrases is in [file: get_prompt.py](https://github.com/Maria-Liakata-NLP-Group/THVAE-summary/blob/main/get_prompt.py), 'read_timeline'.
