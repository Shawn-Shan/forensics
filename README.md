# Traceback of Data Poisoning Attacks in Neural Networks
### ABOUT

This repository contains code implementation of the paper "[Traceback of Data Poisoning Attacks in Neural Networks](https://www.shawnshan.com/files/publication/forensics.pdf)", at *USENIX Security 2022*. 
This traceback tool for poison data is developed by researchers at [SANDLab](https://sandlab.cs.uchicago.edu/), University of Chicago.  

### Note
The code base currently only support CIFAR10 dataset and BadNets attack. Adapting to new dataset is possible by editing the data loading and model loading code. 

### DEPENDENCIES

Our code is implemented and tested on Keras with TensorFlow backend. Following packages are used by our code.

- `keras==2.4.3`
- `numpy==1.19.5`
- `tensorflow-gpu==2.4.1`

Our code is tested on `Python 3.6.8`

### Steps to train a backdoored model and run traceback

#### Step 1: Train/download backdoor model



### Citation
```
@inproceedings{shan2021traceback,
  title={Traceback of Data Poisoning Attacks in Neural Networks},
  author={Shan, Shawn and Bhagoji, Arjun Nitin and Zheng, Haitao and Zhao, Ben Y},
  journal={Proc. of USENIX Security},
  year={2022}
}
```
