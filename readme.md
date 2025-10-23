

# Knee-Deep in C-RASP: A Transformer Depth Hierarchy

This repository is the official implementation of the code for the paper **Knee-Deep in C-RASP: A Transformer Depth Hierarchy**


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>The  only major requirement is torch and tqdm for the main training loop 

## Repository Structure
- 1_layer
	- models
		- L3
			- training_log.txt
			- transformer_model_dim_64_heads_1_layers_9.pth
			> when trained, the output of the 1 layer transformer trained on language L3 will be placed in this folder. There is a training_log.txt file as well as the saved model
		- L4
		- L5
		- ...
- 2_layer
- ...
- configs            
	- config_layer_1.json    
	- config_layer_2.json
	- ...
	> running ./make_configs.sh will generate config files for all the desired transformer configurations. The script can be edited for different setups
- data                
	- L3
		- 0_to_50_src.txt
		- 0_to_50_tgt.txt
		- 50_to_100_src.txt
		- 50_to_100_src.txt
		- ...
		> running generate_data.py will generate source and target pairs each language, sampled uniformly. The specific parameters can be tweaked in the script
- make_configs.sh         
- process_logs.sh         
- reset.sh    
- parse_logs.py      
- generate_data.py            
- train.py    
- readme.md 
- requirements.txt    
- combined_results.csv             

## Description of Scripts

make_configs.sh
- generates config files for each training loop
- you can specify how many heads, epochs, dimensions for each hyperparameter search
- you can specify which training and test files are used for each language

process_logs.sh    
- parses the training_log.txt files from all subfolders and aggregates them into combined_results.csv
 
generate_data.py      
- script to generate data by sampling strings from L_k for each k. 
- the specific quantities can be specified within the file

parse_logs.py  
- utility used by process_logs.sh to parse all training_log.txt files

train.py    
- the main training loop which takes in a config.json file for parameters

 reset.sh 
 - deletes all training_log.txt and transformer models saved

## Training

To train the model(s) in the paper, run this command, replacing configs/config_layer_1.json
with your desired config file

```train
python3 train.py --config configs/config_layer_1.json
```

Make sure to run it in the top level repository so it places the results in the correct subfolders

## Disclaimers

Large Language Models were used to assist the implementing of the training loop and parsing scripts. The code was then manually inspected, tested, and revised. 