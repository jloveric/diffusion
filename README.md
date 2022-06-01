# diffusion
Consider these notes and then implementation stolen from others.  My goal here is to understand this enough and extend to high order methods.

First source was this blog post [diffusion models for machine learning introduction](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)

Experiments in NN diffusion models

Overtrained network which is memorizing test set 151,000 iterations.  Trained on CIFAR10 test images of airplanes on 8GB 2080GPU
Current results are entirely from the above introduction running on a different dataset...  Planning some additional work here and trying on high order networks.

![Predicted samples from cifar10 test set (1000 images of airplanes)](images/sample-151.png "Title")


This paper (the original) is best for understanding why one can make all the assumptions made [Deep Unsupervised Learning Using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf) and the [Theano implementation](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models) from the authors. 

Another useful [writeup](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da) and corresponding [implementation](https://github.com/azad-academy/denoising-diffusion-model)

A [tutorial](https://github.com/acids-ircam/diffusion_models) which goes into everything including historical details. 

Papers
[Generative Modeling by Estimating Gradients of the Data Distribution (Noise Conditional Score Network)](https://arxiv.org/pdf/1907.05600.pdf)

## Training
```
python examples/simple.py
```