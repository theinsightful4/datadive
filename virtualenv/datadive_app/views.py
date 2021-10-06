from django.shortcuts import render
from .modules.datasets import gaussian, spiral # all
from .modules.graphs import scatter

# index page view
def index(request):
    return render(request, 'index.html', 
                  {'gaussian' : scatter.showGaussian(),
                   'spiral' : scatter.showSpiral(),
                   'circle' : scatter.showCircle()})

def result(request):
    return render(request, 'result.html',
                  {'gaussian' : scatter.showGaussian(),
                   'spiral' : scatter.showSpiral(),
                   'circle' : scatter.showCircle()})


