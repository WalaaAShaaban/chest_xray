import os
import cv2

class PrepareData:
    
    def getImages(path):
        image_lst = []
        for img in os.listdir(path):
            image = cv2.imread(path + "/" + img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            image = image/255
            image_lst.append(image)
        return image_lst