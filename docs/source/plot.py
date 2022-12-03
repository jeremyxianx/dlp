import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from itertools import chain
from collections import Counter



def plot_pie_chart(data, labels, epoch):
    
    """
    This function generates the pie chart plot for input data.
    
    
    Parameters
    -------------
    data : array-like of shape (n_samples, n_features)
            Data of interest
    Label : float
            Target vector of data.
 
    epoch : int
            Running epoch 
        
    
    Returns
    -------------
    plot: figure
          Pie chart plot
        
    """
    
    cmap = plt.get_cmap("tab20c")
    colors = cmap(np.array([1,3,5,7,9,10,11,12,14,16,18]))


    porcent = 100.* data / data.sum()
    
    patches, texts = plt.pie(data, colors=colors, startangle=90, radius=1.2)

    centre_circle = plt.Circle((0,0),0.70,fc='white')
    
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    labels = np.array( [output_mapping[element] for element in labels] )
    
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, porcent)]
    
    
    sort_legend = True
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels, data),
                                          key=lambda labels: labels[2],
                                          reverse=True))

    plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-0.25,1.53), ncol = 2, shadow = True,fontsize=13)
    plt.title('Selected backdoor samples'.format(epoch))
    plt.show()