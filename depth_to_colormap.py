import glob
import os

from PIL import Image
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    files = list(glob.glob('evaluate-depths/predicted-*.png'))
    files.extend(list(glob.glob('evaluate-depths/orig-depth-*.png')))
    files.extend(list(glob.glob('evaluate-depths/gt-depth-*.png')))
    for file in files:
        im = np.array(Image.open(file))
        colored_im = np.copy(im)
        # if 'orig-' in file:
        #     im[im == 0] = 255
        if 'gt-' in file:
            colored_im[colored_im == 5] = 255
        elif 'predicted-' in file:
            colored_im[colored_im < 4] = 255
        colored_im = Image.fromarray(np.uint8(cm.jet_r(colored_im) * 255.0))
        colored_im.save('evaluate-depths/colored-'+os.path.basename(file)+'.png')

    fig = plt.figure(figsize=(1, 3))
    ax1 = fig.add_axes([0, 0.05, 0.2, 0.9])
    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=50, vmax=0)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('distance [m]')
    plt.savefig('evaluate-depths/colorbar-cropped.png')

    fig = plt.figure(figsize=(1, 3))
    ax1 = fig.add_axes([0, 0.05, 0.2, 0.9])
    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=1000, vmax=0)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('distance [m]')
    plt.savefig('evaluate-depths/colorbar-orig.png')
