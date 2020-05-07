import cv2
from C_Recognize.Inception_model import InceptionResNetV1
from C_Recognize.Inception_utils import *
import DataSetLink as DLSet
from A_PreProcess.DataProxy import DataProxy
from Tools import *


class RecognizeProxy:
    def __init__(self, threshold):
        self.model = InceptionResNetV1()
        self.model.load_weights(DLSet.Recognizer_link + '/facenet.h5')
        self.threshold = threshold

    def get_vec(self, img):
        return calc_128_vec(self.model, img)

    def get_distance(self, img1, img2, flag=False):
        if flag:
            return face_distance(img1, img2)
        return face_distance(self.get_vec(img1), self.get_vec(img2))


if __name__ == "__main__":
    rp = RecognizeProxy(threshold=1)
    data_proxy = DataProxy(DLSet.train_data_link, DLSet.train_label_link)
    X, y = data_proxy.get_batch(2, 5, random_flag=True)
    print()
    print(face_distance(rp.get_vec(X[0, 0]), rp.get_vec(X[1, 1])))

