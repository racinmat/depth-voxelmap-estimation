import glob
import os

from PIL import Image
from matplotlib import cm
import numpy as np

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
        colored_im = Image.fromarray(np.uint8(cm.jet(255 - colored_im) * 255.0))
        colored_im.save('evaluate-depths/colored-'+os.path.basename(file)+'.png')
