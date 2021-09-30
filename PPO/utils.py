import visdom
import numpy as np


class DrawLine():

    def __init__(self, env, title, xlabel=None, ylabel=None):
        self.vis = visdom.Visdom()
        self.update_flag = False
        self.env = env
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def __call__(
            self,
            xdata,
            ydata,
    ):
        if not self.update_flag:
            self.win = self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                opts=dict(
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    title=self.title,
                ),
                env=self.env,
            )
            self.update_flag = True
        else:
            self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                win=self.win,
                env=self.env,
                update='append',
            )
import numpy as np
from PIL import Image
import torchvision

def showImage(imageArray):
    # print(imageArray)
    img = Image.fromarray(imageArray, 'RGB')
    img.show()

def saveImage(imageArray, name="default"):
    img = Image.fromarray(imageArray, 'RGB')
    img.save('image1.png')

def showTensor(tensor):
    img = torchvision.transforms.ToPILImage()(tensor)
    img.show()

def saveTensor(tensor):
    img = torchvision.transforms.ToPILImage()(tensor)
    img.save('img/tensor1.png')

def fancyPreprocess(image):
    for i in range(len(image)):
        mask = np.array([107, 107, 107])
        print(mask)
        image[i][mask] = [255, 0, 255]
    return image

def standardPreprocess(image):
    image = np.mean(image[:, :], axis=2)
    # image = image[:2, :2]
    return image