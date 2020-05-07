import os
import sys
sys.path.append('../')


""" ------------------------- base setting -----------------------------"""
dataset_link = '../../DataSet'
dataset_name = ['LFW']
dataset_choice = dataset_name[0]


""" ------------------------ folder set ------------------------- """
mnist_link = dataset_link + '/mnist.npz'
base_link = dataset_link + '/%s' % dataset_choice

Detector_link = dataset_link + '/Detector'
Recognizer_link = dataset_link + '/Recognizer'
RawSet_link = base_link + '/RawSet'
CleanSet_link = base_link + '/CleanSet'
MainSet_link = base_link + '/MainSet'
ModelSet_link = base_link + '/ModelSet'


""" ------------------------ raw set ------------------------- """
raw_data_link = RawSet_link + '/Faces'


""" ------------------------ clean set ------------------------- """
train_data_link = CleanSet_link + '/train_data'
train_label_link = CleanSet_link + '/train_label'
test_data_link = CleanSet_link + '/test_data'
test_label_link = CleanSet_link + '/test_label'
map_label2id_link = CleanSet_link + '/map_label2id'


""" ------------------------ main set ------------------------- """


""" ------------------------ model set ------------------------- """
model_save_link = ModelSet_link + '/%s_%s.h5' % ('%s', dataset_choice)
history_save_link = ModelSet_link + '/history_%s_%s' % ('%s', dataset_choice)
cmc_res_link = ModelSet_link + '/cmc_result_%s_%s' % ('%s', dataset_choice)
roc_res_link = ModelSet_link + '/roc_result_%s_%s' % ('%s', dataset_choice)

""" ------------------------ app set ------------------------- """
AppSet_link = dataset_link + '/App'
