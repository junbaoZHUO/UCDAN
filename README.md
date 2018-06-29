# UCDAN<br>
Tensorflow implementation of UCDAN<br>

## Prepare data <br>
cd data/<br>
tar -zxvf amazon.tar.gz<br>
tar -zxvf webcam.tar.gz<br>
tar -zxvf dslr.tar.gz<br>

## Prepare preptrained models<br>
Get AlexNet pretrained model via http://www.cs.toronto.edu/%7Eguerzhoy/tf_alexnet/bvlc_alexnet.npy<br>
Get ResNet-50 pretrained model via http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz<br>

## Traing scripts<br>
For AlexNet<br>
cd AlexNet<br>
CUDA_VISIBLE_DEVICES=0 python train.py --src amazon --tar webcam --epoches 60 --lr0 0.0001 --bs 128 --num_class 31 --weight_decay 0.00001 --group_or_not True --gp_weight 6.5 --gap_weight 3 --fc_weight 1.5 2>&1 | tee AW_GR_6.5_GAP_3_FC_1.5.log<br>

For ResNet<br>
cd ResNet<br>
CUDA_VISIBLE_DEVICES=0 python train.py --src amazon --tar webcam --epoches 40 --lr0 0.0001 --bs 32 --num_class 31 --weight_decay 0.00005 --group_or_not True --gp_weight 1 --gap_weight 0.8 --fc_weight 0.8 2>&1 | tee LOG/AW_GR_1.0_GAP_0.8_FC_1.0.log<br>
CUDA_VISIBLE_DEVICES=0 python test.py  --src amazon --tar webcam --epoches 40 --bs 15 --num_class 31<br>
## Citation<br>
If you use this library for your research, we would be pleased if you cite the following papers:<br>
@inproceedings{Zhuo:2017:DUC:3123266.3123292,<br>
 author = {Zhuo, Junbao and Wang, Shuhui and Zhang, Weigang and Huang, Qingming},<br>
 title = {Deep Unsupervised Convolutional Domain Adaptation},<br>
 booktitle = {Proceedings of the 2017 ACM on Multimedia Conference},<br>
 series = {MM '17},<br>
 year = {2017},<br>
 isbn = {978-1-4503-4906-2},<br>
 location = {Mountain View, California, USA},<br>
 pages = {261--269},<br>
 numpages = {9},<br>
 url = {http://doi.acm.org/10.1145/3123266.3123292},<br>
 doi = {10.1145/3123266.3123292},<br>
 acmid = {3123292},<br>
 publisher = {ACM},<br>
 address = {New York, NY, USA},<br>
 keywords = {attention model, correlation alignment, deep learning, unsupervised domain adaptation},<br>
} <br>

