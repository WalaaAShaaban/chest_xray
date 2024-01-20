import os
import cv2

class PrepareData():
    
    def getImages(path):
        image_lst = []
        for img in os.listdir(path):
            image = cv2.imread(path + "/" + img)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (225, 225))
            image = image/255
            image_lst.append(image)
        return image_lst