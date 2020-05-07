import sys; sys.path.append('../')
from Tools import *
import DataSetLink as DLSet


# node with all the images of label
class Node:
    def __init__(self):
        self.len = 0
        self.images = []

    # add a new image
    def add(self, image):
        self.len += 1
        self.images.append(image)

    # choose n images by random
    def choose(self, n):
        return random.sample(self.images, n)


# data proxy
class DataProxy:
    # init the data proxy by generate a dict contains all the {label: Node[image1, images2, ...]}
    def __init__(self, images_link, labels_link):
        # 1. load origin data
        images_org = load_obj(images_link)
        labels_org = load_obj(labels_link)

        # 2. build dict
        self.data = {}
        for i in range(len(labels_org)):
            if labels_org[i] not in self.data.keys():
                self.data[labels_org[i]] = Node()
            self.data[labels_org[i]].add(images_org[i])

        # 3. information of proxy
        self.keys = list(self.data.keys())
        self.total = len(self.keys)
        self.start = 0                          # loop for data generate for training

    # get a batch of data
    def get_batch(self, batch_size, np_ratio, random_flag=False):
        # 1. choose #batch_size different images
        major = []
        # batch_size decides choose all data or just the number of batch_size
        if batch_size != -1:
            while len(major) < batch_size:
                # random_flag decides whether choose random by loop
                if random_flag:
                    label = random.choice(self.keys)
                    if self.data[label].len >= 2:
                        major.append(label)
                else:
                    if self.data[self.keys[self.start]].len >= 2:
                        major.append(self.keys[self.start])
                    self.start = (self.start + 1) % self.total
        else:
            for label in self.keys:
                if self.data[label].len >= 2:
                    major.append(label)
        
        # 2. pack up
        pairs = []
        signs = []
        for each in major:
            # 2.1 add the same label's images
            same_images_path = self.data[each].images
            same_images = []
            for i in range(len(same_images_path)):
                same_images.append(image2matrix(same_images_path[i]))

            # 2.2 add np_ratio diffs
            n_same = (len(same_images) - 1) if random_flag else 1
            for i in range(n_same):
                pairs.append([same_images[i], same_images[i+1]])
                signs.append([1])
                for _ in range(np_ratio):
                    pairs.append([same_images[i],
                                  image2matrix(self.data[random_choose_one(self.keys, [each])].choose(1)[0])])
                    signs.append([0])

        pairs = np.array(pairs)
        signs = np.array(signs).astype(float)
        return pairs, signs


if __name__ == '__main__':
    data_proxy = DataProxy(DLSet.train_data_link, DLSet.train_label_link)
    X, y = data_proxy.get_batch(1, 2)
    print(X[0, 0].shape)
    print(X[0, 1].shape)
    print(X[1, 0].shape)
    print(X[1, 1].shape)
    print(X[2, 0].shape)
    print(X[2, 1].shape)
    print(X.shape)
    show_img(matrix2image(X[0, 0]))
    show_img(matrix2image(X[0, 1]))
    show_img(matrix2image(X[1, 0]))
    show_img(matrix2image(X[1, 1]))
    show_img(matrix2image(X[2, 0]))
    show_img(matrix2image(X[2, 1]))
