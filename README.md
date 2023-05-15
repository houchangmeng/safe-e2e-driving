# End-to-end Autonomous Driving With Safety Constraints

[[code]](https://github.com/houchangmeng?tab=repositories)  
[[paper]](www.baidu.com)

This is a PyTorch implementation of "End-to-end Autonomous Driving with Safety constraint" proposed in:

- ["End-to-end Safe Autonomous Driving with Safety constraint" by Hou, Changmeng, and Zhang, Wei.](https://openreview.net)  
  
## Requirement

Ubuntu 20.04  
Carla 0.9.13  
Python 3.7.12  

## Installation

1. Set up conda environment.
```
$ conda create -n env_name python=3.7
$ conda activate env_name
```
2. Clone this git repo to an appropriate folder.
```
$ git clone https://github.com ....
```
3. Enter the root folder of this repo and install the packages.
```
$ pip install -r requirements.txt
```
4. Train the safe agent .
```
$ python train_safe.py
```

## Result

Results on custom carla reinforcement environment are as follows.

<img src="./images/learning_curve.png" height="512" width="484">

### Ramp merging  
<img src="./images/safe-merge.gif" height="256" width="384">

### Roundabout  
<img src="./images/safe-roundabout.gif" height="256" width="384">

### Intersection  
<img src="./images/safe-intersection.gif" height="256" width="384">

- The first row shows the **ground truth** of camera, lidar, and semantic birdeye images when the trained agent is running in the simulated town with surrounding vehicles and walkers (green boxes).
- The second row shows the **reconstructed** camera, lidar, and semantic birdeye images from the latent state, which is inferred online with the learned sequential latent model.

## Sample a sequence

<img src="./images/sample-latent.png" >

Sample a sequence of camera, lidar and birdeye images.

- The first row shows the ground truth.
- The secend row shows the posterior sample.
- The third row shows the conditional prior sample.
- The forth row shows the prior sample.
- Left to right indicates flowing time steps.

# Reference
[[1]](https://openreview.net/forum?id=b39dQt_uffW)Yannick Hogewind, Thiago D. Simão, Tal Kachman, and Nils Jansen. "Safe Reinforcement Learning From Pixels Using a Stochastic Latent Representation" , ICLR 2023.  
[[2]](https://arxiv.org/abs/2001.08726) Chen, Jianyu and Li, Shengbo Eben and Tomizuka, Masayoshi."Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning" arXiv preprint arXiv:2001.08726(2020).

