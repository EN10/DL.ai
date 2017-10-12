import os  
import urllib
file = urllib.URLopener()

if not os.path.isfile('train-images-idx3-ubyte.gz'):
    file.retrieve("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz")    
if not os.path.isfile('train-labels-idx1-ubyte.gz'):
    file.retrieve("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz")
if not os.path.isfile('t10k-images-idx3-ubyte.gz'):
    file.retrieve("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz")
if not os.path.isfile('t10k-labels-idx1-ubyte.gz'):
    file.retrieve("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

import gzip
with gzip.open("train-images-idx3-ubyte.gz", "rb") as f:
    file_content = f.read()
    print file_content