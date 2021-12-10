import os
import scipy.ndimage
import scipy.misc
import skimage.io
from skimage import img_as_ubyte

from pdb import set_trace as st 

base="/data/datasets/trojai/trojai-round0-dataset/id-00000009/example_data"

savedir="/nethome/sunmj/trojai-example/data/id-9"

for class_id in [0,1,2,3,4]:
    for i in range(20):
        filename = "class_%d_example_%d.png"%(class_id, i)
        save_path = os.path.join(savedir, "train", str(class_id))

        img_path = os.path.join(base, filename)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img = skimage.io.imread(img_path)
        skimage.io.imsave(os.path.join(save_path, filename), img_as_ubyte(img))

    for i in range(20, 100):
        filename = "class_%d_example_%d.png"%(class_id, i)
        save_path = os.path.join(savedir, "test", str(class_id))

        img_path = os.path.join(base, filename)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img = skimage.io.imread(img_path)
        skimage.io.imsave(os.path.join(save_path, filename), img_as_ubyte(img))