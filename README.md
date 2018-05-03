# UCDAN
Tensorflow implementation of UCDAN

## Prepare data 
cd data/

tar -zxvf amazon.tar.gz

tar -zxvf webcam.tar.gz

tar -zxvf dslr.tar.gz

## Prepare preptrained models
Get AlexNet pretrained model via http://www.cs.toronto.edu/%7Eguerzhoy/tf_alexnet/bvlc_alexnet.npy

Get ResNet-50 pretrained model via http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz

## Citation
If you use this library for your research, we would be pleased if you cite the following papers:

@inproceedings{Zhuo:2017:DUC:3123266.3123292,
 author = {Zhuo, Junbao and Wang, Shuhui and Zhang, Weigang and Huang, Qingming},
 title = {Deep Unsupervised Convolutional Domain Adaptation},
 booktitle = {Proceedings of the 2017 ACM on Multimedia Conference},
 series = {MM '17},
 year = {2017},
 isbn = {978-1-4503-4906-2},
 location = {Mountain View, California, USA},
 pages = {261--269},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3123266.3123292},
 doi = {10.1145/3123266.3123292},
 acmid = {3123292},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {attention model, correlation alignment, deep learning, unsupervised domain adaptation},
} 
