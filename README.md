# CIFAR-ResNet20
Implementation of a ResNet20 for classifying CIFAR-10 samples.

## Instructions
1. Open Google Colab: https://research.google.com/colaboratory/
2. Upload and open: colab.ipynb
3. Request GPU (Runtime -> Change runtime type -> Hardware accelerator: GPU)
4. Upload: env.yml, cifar_io.py, resnet20.py
5. Run colab.ipynb cell by cell

## Motivation
The goal of this project was to re-implement the ResNet20 architecture for classifying CIFAR-10 samples. The original approach and network architecture is descibed in:
![He et al., Deep Residual Learning for Image Recognition, 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

## Results
The trained ResNet20 yields a classification accuracy of XX.XX% on test data, which is a meaningful improvement compared to a plug-and-play SVM classification approach, which gives a classification accuracy of XX.XX% on the same test data.

The trained ResNet20 yields a classification accuracy of XX.XX% on test data, which is very close to the accuracy stated by He et al., i.e. 91.25%.

![alt text](https://github.com/arnemonsees/cifar-resnet20/blob/main/sample.png)
