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
Go to here and download the training, validation, and test point clouds: https://shapenet.cs.stanford.edu/iccv17/

For some reason the shell script will not download the dataset:
~~For linux using wget (adjust using curl for mac)

```
bash download.sh
```
~~

## Information about the dataset labeling
In the train, validation, and test datasets there are folders labeled by numbers that represent the specific object class. Here is what these numbers mean:

```
Airplane        02691156
Bag             02773838
Cap             02954340
Car             02958343
Chair           03001627
Earphone        03261776
Guitar          03467517
Knife           03624134
Lamp            03636649
Laptop          03642806
Motorbike       03790512
Mug             03797390
Pistol          03948459
Rocket          04099429
Skateboard      04225987
Table           04379243
```




