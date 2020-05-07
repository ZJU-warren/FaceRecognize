from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import sys; sys.path.append('../')
import pathlib
from sklearn.model_selection import train_test_split
import DataSetLink as DLSet
from Tools import *
from B_Detect_Align.DetectProxy import DetectProxy
# import cv2


# generate data like this form [(img_path, img_label), ...]
def generate_data(data_link, test_size=0.3, store_link=None):
    dp = DetectProxy()

    folder_path_set = list(pathlib.Path(data_link).iterdir())
    images = []
    labels = []
    label_id_map = []

    label_id = 0
    # visit all the folder
    for folder_path in folder_path_set:

        # obtain the img paths
        label = str(folder_path).split('/')[-1]
        label_id_map.append([label_id, label])

        # load img and append it
        img_path_set = folder_path.iterdir()
        for img_path in img_path_set:
            # detect face
            img = cv2.imread(str(img_path))
            draw = dp.detect(img)
            draw = cv2.resize(draw, (150, 150))
            cv2.imwrite(str(img_path)[:-4] + '_crop.jpg', draw)

            # append
            images.append(str(img_path)[:-4] + '_crop.jpg')
            labels.append(label_id)

        label_id += 1
        if label_id % 100 == 0:
            print('----------------- processed: %d / 1680 -----------------' % label_id)

    if store_link is not None:
        store_obj(label_id_map, store_link)

    images = np.array(images)
    labels = np.array(labels)

    # split the data by test_size
    return train_test_split(images, labels, test_size=test_size)


if __name__ == '__main__':
    train_images, test_images, train_labels, test_labels \
        = generate_data(DLSet.raw_data_link, test_size=0.1, store_link=DLSet.map_label2id_link)
    store_obj(train_images, DLSet.train_data_link)
    store_obj(train_labels, DLSet.train_label_link)
    store_obj(test_images, DLSet.test_data_link)
    store_obj(test_labels, DLSet.test_label_link)
