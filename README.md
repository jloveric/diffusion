# Image Generation using Diffusion Models
Model modified from [this git repo](https://github.com/lucidrains/denoising-diffusion-pytorch) to use pytorch lightning, hydra, tensorboard, nevergrad and poetry.  Intend to extend this to other types of networks including high order networks.

## Training
```
python examples/simple.py
```
## Configuration
Configuration file is in yaml [here](configs/simple.yaml)

## Results
Experiments in NN diffusion models.

![Predicted samples from cifar10 training set](images/sample-151.png)

# References
* [diffusion models for machine learning introduction](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
* [Deep Unsupervised Learning Using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf) and the [Theano implementation](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models) from the authors. 
* [Diffusion models made easy](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da) and corresponding [implementation](https://github.com/azad-academy/denoising-diffusion-model)
* A [tutorial](https://github.com/acids-ircam/diffusion_models) which goes into everything including historical details. 
* [Generative Modeling by Estimating Gradients of the Data Distribution (Noise Conditional Score Network)](https://arxiv.org/pdf/1907.05600.pdf)
* [Denoising Diffusion Probabalistic Models](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
* [Improved Denoising Diffusion Probabalistic Models](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)

