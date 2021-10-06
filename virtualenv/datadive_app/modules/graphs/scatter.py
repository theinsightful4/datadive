import io
import urllib, base64
import matplotlib.pyplot as plt
from ...modules.datasets import gaussian, plane, circle, spiral

def showGaussian():
    X, y = gaussian.createGaussian(n_samples = 120, centers=2,
                  cluster_std=2.0, random_state=1)
    
    f = plt.figure(figsize=(6,5))
    #plt.style.use('grayscale')
    
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    
    plt.scatter(X[:,0], X[:,1], c=y, s=20)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    return uri

def showSpiral():
    X, y = spiral.createSpiral(n_samples=500, noise=0.07, random_state=0)
    
    f = plt.figure(figsize=(6,5))
    #plt.style.use('grayscale')
    
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    
    plt.scatter(X[:,0], X[:,1], c=y, s=25)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    return uri

def showCircle():
    X, y = circle.createCircle(n_samples=120,noise=0.05,random_state=0)
    X1, y1 = circle.createCircle(n_samples=100, noise = 0.58,random_state=0)
    
    f = plt.figure(figsize=(6,5))
    #plt.style.use('grayscale')
    
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    
    plt.scatter(X1[:,0] * 0.20, X1[:,1] * 0.20, s=25)
    plt.scatter(X[:,0], X[:,1], s=25)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    fig = plt.gcf()
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    return uri
    
