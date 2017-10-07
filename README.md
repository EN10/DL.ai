# Neural Networks and Deep Learning

**Logistic Regression with a Neural Network**  
cats.py - Image recognition algorithm that recognizes cats with 70% accuracy!

**Planar data classification with a hidden layer**  
flowers.py - neural network to correctly classify red and blue generated points that form a flower

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