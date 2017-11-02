# Neural Networks and Deep Learning

**w1: Intro**

**w2: Logistic Regression with a Neural Network**  
cats.py - Image recognition algorithm that recognizes cats with 70% accuracy!  
Single neuron of linear and sigmoid.
prop: linear -> sigmoid  

**w3: Planar data classification with a hidden layer**  
flowers.py - neural network to correctly classify red and blue generated points that form a flower  
Testing different numbers of hidden units in a single hidden layer network.  
forward_prop: linear -> tanh -> linear -> sigmoid  

**w4: Deep Neural Network**  
5-layer Neural Network.py - a deep neural network that classifies cat vs. non-cat images.  
L_layer_model prop: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID 

**w5: I.R.G.**  
Initialization [of weights]  
Regularization (L2 & Dropout) [reduces overfitting]  
Gradient Checking [verifies backprop]  

**w6 Optimization**  
Mini-Batch and Adam

**w7: Tensorflow**

### Jupyter

These notebooks have been made for Python3

Install:

    sudo pip3 install jupyter

Run:

    jupyter notebook --ip=0.0.0.0 --port=8080 --no-browser

### Prerequisites:

    sudo apt update
    sudo apt install python-pip3
    
    sudo -H pip3 install --upgrade pip   
    sudo pip3 install numpy h5py
    
### Python2 Compatibility

`/` works differently in Python3 vs Python2  
__future__ seems to fix this:  

    from __future__ import division