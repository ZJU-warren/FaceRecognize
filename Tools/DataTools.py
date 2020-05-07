import random
import pickle
from sklearn.preprocessing import LabelBinarizer


# my label one-hot class
class MyLB:
    def __init__(self, org_label):
        self.lb = LabelBinarizer()
        self.lb.fit_transform(org_label)

    def transform(self, org_label):
        return self.lb.transform(org_label)


# random choose a different
def random_choose_one(all_set, diff_set):
    choice = random.choice(all_set)
    while choice in diff_set:
        choice = random.choice(all_set)
    return choice


# store object
def store_obj(obj, data_link):
    pickle.dump(obj, open(data_link, 'wb'), protocol=4)


# load object
def load_obj(data_link):
    return pickle.load(open(data_link, 'rb'))

