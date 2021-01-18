# POP (Properties Optimization Point)
This repository contains the implementation of the following paper:

```tex
@article{2021UnsupervisedPOP,
  title={Unsupervised Learning Framework for Interest Point Detection and Description via Properties Optimization},
  author={Pei Yan and Yihua Tan and Yuan Tai and Dongrui Wu and Hanbin Luo and Xiaolong Hao},
  journal={Pattern Recognition},
  year={2021},
}
```

## Getting started
You can get the complete code with the git clone command:
```bash
git clone https://github.com/elvintanhust/POP-Interest-Point.git
cd POP-Interest-Point
```
This code is developed with Python 3.8 and PyTorch 1.4, but the packages with the later versions should be also suitable. Typically, conda and pip are the recommended ways to configure the environment:
```bash
conda create -n POP python=3.8
conda activate POP
conda install numpy matplotlib scipy
conda install pytorch cudatoolkit=10.1 -c pytorch
pip install imgaug
pip install 'opencv-contrib-python<=3.12'
```

## Pretrained models
We provide the pre-trained models in the `save_POP_model/` and `save_recon_model/` folders:
* `save_POP_model/POP_net_pretrained.pth` in the folder: the model to achieve the detection and description of interest point. It is the model denoted as `POP` in the paper, which is used in most experiments.
* `save_recon_model/recon_net_pretrained.pth` in the folder: the model of reconstructor which is detailed in the Section 3.4. Here the model is trained with COCO 2014 dataset. During the training of POP, this model is used in the computation of informativeness property. Note POP no longer depends on this model in the testing stage.

## The evaluation of POP
We provide the evaluation code for [HPatches](https://github.com/hpatches/hpatches-dataset) dataset, which references the evaluation process of [SuperPoint](https://github.com/rpautrat/SuperPoint). Before perform the evaluation on the entire HPatches, you can first verify the environment by running the script directly:
```bash
python eval_POP_net.py
```
If the environment is configured correctly, the following information will be printed in the terminal:
```text
--------start the evaluation of POP--------
(epsilon=1) id:0 repeat:x.xxx homo_corr:x.xxx m_score:x.xxx ...
## accuracy on the entire dataset (epsilon=1):
## repeat:x.xxxxx homo_corr:x.xxxxx m_score:x.xxxxx ...
(epsilon=3) id:0 repeat:x.xxx homo_corr:x.xxx m_score:x.xxx ...
## accuracy on the entire dataset (epsilon=3):
## repeat:x.xxxxx homo_corr:x.xxxxx m_score:x.xxxxx ...

--------start the evaluation of superpoint--------
...
```
Here POP and several comparison methods ([SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), SIFT, ORB) are evaluated on the `i_ajuntament` sequence in HPatches. The default model of POP is set as `save_POP_model/POP_net_pretrained.pth`. Note the `i_ajuntament` sequence has been placed in the `hpatches-sequences-release` folder so that the `eval_POP_net.py` script can be run directly. Furthermore, the pre-trained model `superpoint_v1.pth` provided by [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) is also included in this repository to simplify the configuration.

In the above process, the `statistics_results` folder is created automatically and the main statistics results are written in it. After the evaluations of all methods, four text files, namely `ORB.txt`, `POP_net.txt`,  `SIFT.txt`,  `superpoint.txt`, should appear in this folder. 

The environment is verified to be correct if the above statistics results can be outputted. Then you can place other data in the `hpatches-sequences-release` folder to further evaluate the methods. Note the format of the data should be consistent to the HPatches sequences. One convenient way is to download the [HPatches](https://github.com/hpatches/hpatches-dataset) sequences first, and then unzip it into the `hpatches-sequences-release` folder.

For more details about all parameters of `eval_POP_net.py`, run `python eval_POP_net.py --help`.

## Training the model
### Training POP without informativeness property
You can first verify the environment by running the script directly:
```bash
python train_POP_net.py
```
This command performs the training process of POP, and the training images are in the `demo_input_images/` folder. If the environment is configured correctly, the following information will be printed in the terminal:
```text
## ep:1 iter:0/1 ##  loss_detection:x.xxx  loss_description:x.xxxx  loss_informativeness:x.xxxx ...
## ep:3 iter:0/1 ##  loss_detection:x.xxx  loss_description:x.xxxx  loss_informativeness:x.xxxx ...
```
And the training process will write the checkpoints of POP into the `save_POP_model/` folder. The name of the checkpoint is formatted as `POP_net_epoch_endx`.

The environment is verified to be correct if the above process can be finished without error. To train your model, you can place other data in the `demo_input_images` folder, or specify the training folder with the `--train-image-path` parameter: 
```bash
python train_POP_net.py --train-image-path /the/path/of/training/dataset
```
To reproduce the performance in the paper, you can train POP with [COCO 2014 training set](http://images.cocodataset.org/zips/train2014.zip) (containing 82783 images). In our experiments, one or two epochs are normally enough to achieve the performance similar to that in the paper.

You can also set the `--POP-checkpoint-path` parameter to make the model be initialized with the given checkpoint:

```bash
python train_POP_net.py --POP-checkpoint-path /the/path/of/POP/checkpoint
```
For more details about all parameters of `train_POP_net.py`, run `python train_POP_net.py --help`.

### Training POP with informativeness property
The computation of informativeness depends on the model of reconstructor which is detailed in the Section 3.4. We have provided the pre-trained model of reconstructor, namely `recon_net_pretrained.pth` in the `save_recon_model/` folder. So you can train POP with informativeness property by running
```bash
python train_POP_net.py --recon-net-path save_recon_model/recon_net_pretrained.pth
```

### Training the model of reconstructor

You may want to train the model of reconstructor which is detailed in the Section 3.4. To achieve it, you just need to place your training images in the `demo_input_images/` folder and then run the script:

```bash
python train_recon_net.py
```
Or you can specify the training folder with the `--train-image-path` parameter: 
```bash
python train_recon_net.py --train-image-path /the/path/of/training/dataset
```
You can also set the `--recon-checkpoint-path` parameter to make the model initialized with the checkpoint:
```bash
python train_POP_net.py --recon-checkpoint-path /the/path/of/reconstructor/checkpoint
```
During the training process, the following information will be printed in the terminal:

```bash
ep:1 iter:0/1  loss:x.xxxx
ep:3 iter:0/1  loss:x.xxxx
```
And the training process will write the checkpoints of the reconstructor into the `save_recon_model/` folder. The name of the checkpoint is formatted as `recon_net_epoch_endx`.

For more details about all parameters of `train_recon_net.py`, run `python train_recon_net.py --help`.

## The visualization of informativeness results

In the Section 5.5 of paper, we visualize the informativeness results in Fig. 3. This can be achieved by running:
```bash
python recon_full_image.py
```
Then the images `demo_input_images/` folder are considered as the input images, and the  the results can be obtained in the `recon_image_results/` folder. Note the visualization results here are slightly different from that shown in the paper. The reason is that here the pre-trained reconstructor model is just trained with COCO dataset but not fineturned for the single image.

For more details about all parameters of `recon_full_image.py`, run `python recon_full_image.py --help`.