import matplotlib.pyplot as plt

"""
   Helper function for plotting two images. 
"""
def plot2(image1, image2, figsize=(14, 9), title1="", title2="", cmap1=None, cmap2=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(title1)
    ax2.imshow(image2, cmap=cmap2)
    ax2.set_title(title2)
    plt.show()
