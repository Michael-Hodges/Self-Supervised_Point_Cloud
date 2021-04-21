# Self-Supervised_Point_Cloud
This repo is code for elf-Supervised Deep Learning on Point Clouds by Reconstructing Space as can be found here: https://papers.nips.cc/paper/2019/file/993edc98ca87f7e08494eec37fa836f7-Paper.pdf 

## What this repo does
This repo takes the ShapeNet dataset and produces a permutated and labeled data set and example is shown below: 

![alt text](https://github.com/Michael-Hodges/Self-Supervised_Point_Cloud/blob/main/images/pre_trans.png?raw=true)
![alt text](https://github.com/Michael-Hodges/Self-Supervised_Point_Cloud/blob/main/images/post_trans.png?raw=true)

Algorithm from the paper: 
![alt text](https://github.com/Michael-Hodges/Self-Supervised_Point_Cloud/blob/main/images/algorithm.png?raw=true)

## What this repo doesn't do
It does not have any deep learning models avalailable to be trained as is done in the paper it just creates the dataset.

## Installing Dependencies and downloading dataset:
Requirements are as follows:
- numpy
- pptk (for visualization)

**Download dataset as follows:**

For linux using wget (adjust using curl for mac)
'''
bash download.sh
'''



