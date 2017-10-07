# Logistic Regression with a Neural Network mindset

cats.py - Image recognition algorithm that recognizes cats with 70% accuracy!

Based on Neural Networks and Deep Learning by deeplearning.ai

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
    
### Python2

`/` works differently in Python3 vs Python2  
__future__ seems to fix this:  

    from __future__ import division