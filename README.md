## Code for paper "Continual Adversarial Defense". 

## Get Started

Datasets are CIFAR10.

Our codebase accesses the datasets from `./data/` and checkpoints from `./net_weights/` by default.
```
├── ...
├── data
│   
├── net_weights
│   
├── cifar10_online.py
├── ...
```

All of the adversarial data are generated using torchattacks.
Please configure config_cifar10.py first.


### Data
Our data is converted to .pt formation. You can make adversarial data using make_adv_normal.py.


### Pretrained Model
You can download pretrained clean model from [here](https://huggingface.co/cc13qq/cifar10_wrn-28-10/tree/main).
And put it to the direction './net_weights/Clean/wrn-28-10-dropout0.3.pth'.

### Run
python cifar10_online.py 


```
## Dependencies
python 3.8.8, PyTorch = 1.10.0, cudatoolkit = 11.7, torchattack, torchvision, tqdm, scikit-learn, mmcv, numpy, opencv-python, dlib, Pillow
```
