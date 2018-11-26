from PIL import Image
import numpy as np


class ImageInspector:
    def __init__(self, img_path):
        self.img = Image.open(img_path)
        
    def display(self):
        self.img.show()

    def display_as_array(self):
        arr = np.asarray(self.img, dtype="int32")
        print(arr)

    def find_unique_numbers(self):
        arr = np.asarray(self.img, dtype="int32")
        print(np.unique(arr))
        
        
ii = ImageInspector("./images/000000000009.png")
ii.find_unique_numbers()

