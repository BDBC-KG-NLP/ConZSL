# ConZSL
The code and data for the paper "Prototype-Free Contrastive Learning for Generalized Zero-Shot Object Recognition (Under Review)".

# Intstall Requirements

pip install -r requirements.txt



# Datasets

We provide our data in the following anonymous [Baidu Cloud link](https://pan.baidu.com/s/1gRMFQ9LDL4uCJrx_Z-3q6w) Access Code: m1i4.

All processed datasets are in the "data" folder.



# Model Training

We provide the shell script "run.sh" for training ConZSL on CUB, 
and the shell script "run_all.sh" is to train all models on all datasets.

Use command "sh run.sh" to train the model.



# Model Test

We provide the test script "test.sh" for ConZSL on CUB. 

Use command "sh test.sh" can directly evaluate ConZSL on CUB based 

For other models and datasets, first run the training script to train the model, 
and append "--test" command to the training script for testing. 
