# SSDS
The source code for our paper: 
Symmetrical Self-representation and Data-grouping Strategy for Unsupervised Feature Selection

## Function File
File SSDS.m is the function of our method to call from Matlab.

## Steps to reproduce the experimental results

File StepsToReproduce.mlx contains the steps to generate the experiment results, and all needed support function.
The dataset mentioned in the code can be found in the following link: https://jundongl.github.io/scikit-feature/datasets.html

### Steps:
0. Put File StepsToReproduce.mlx to the working directory of Matlab.
1. Put the data of .mat files to the sub-folder of './datasets/'. Change accordingly the dataset file name in the variable of "datalist";
2. Change the value of parameters or the range of parameters, if needed. (In the 1st code section) The parameters are alpha, beta, gamma, sigma and the number of groups (group_num)
3. Press Run button to have a look at the result output. This main file test all parameters from the pre-defined ranges one by one.
