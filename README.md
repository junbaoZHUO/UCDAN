# UCDAN
Tensorflow implementation of UCDAN

## Prepare data 
cd data/\<br> 
tar -zxvf amazon.tar.gz\<br> 
tar -zxvf webcam.tar.gz\<br> 
tar -zxvf dslr.tar.gz

## Prepare preptrained models
Get AlexNet pretrained model via http://www.cs.toronto.edu/%7Eguerzhoy/tf_alexnet/bvlc_alexnet.npy\<br> 
Get ResNet-50 pretrained model via http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz

## Citation
If you use this library for your research, we would be pleased if you cite the following papers:\<br> 

@inproceedings{Zhuo:2017:DUC:3123266.3123292,\<br> 
 author = {Zhuo, Junbao and Wang, Shuhui and Zhang, Weigang and Huang, Qingming},\<br> 
 title = {Deep Unsupervised Convolutional Domain Adaptation},\<br> 
 booktitle = {Proceedings of the 2017 ACM on Multimedia Conference},\<br> 
 series = {MM '17},\<br> 
 year = {2017},\<br> 
 isbn = {978-1-4503-4906-2},\<br> 
 location = {Mountain View, California, USA},\<br> 
 pages = {261--269},\<br> 
 numpages = {9},\<br> 
 url = {http://doi.acm.org/10.1145/3123266.3123292},\<br> 
 doi = {10.1145/3123266.3123292},\<br> 
 acmid = {3123292},\<br> 
 publisher = {ACM},\<br> 
 address = {New York, NY, USA},\<br> 
 keywords = {attention model, correlation alignment, deep learning, unsupervised domain adaptation},\<br> 
} 
