import pickle
import random

open_file = open("sample.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()

categorise_list = [list(), list(), list(), list(), list(), list()]

for dct in loaded_list:
    categorise_list[dct['output']].append(dct)

final_list = list()

for i in range(1,6,1):
    final_list = final_list + random.sample(categorise_list[i], len(categorise_list[5]))

open_file = open("balanced_class_data_10x10_250000.pkl", "wb")
pickle.dump(final_list, open_file)
open_file.close()
