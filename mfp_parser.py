import pandas as pd

from glob import glob
from os.path import join, split

file_list = glob(join('food101', '*'))
nutri_dict_list = list()
for file_name in file_list:
    nutri_dict = dict()
    file_handle = open(file_name, 'r')
    _, food_label = split(file_name)
    food_label = food_label.split('.')[0]
    block_flag = False
    for line in file_handle:
        if 'col-1' in line and 'alorie' in line:
            block_flag = True
        elif block_flag:
            calorie_count = int(line.split('>')[1].split('<')[0].replace(',', ''))
            block_flag = False
    nutri_dict['label'] = food_label
    nutri_dict['calories'] = calorie_count
    nutri_dict_list.append(nutri_dict)
nutri_df = pd.DataFrame(nutri_dict_list)
nutri_df.to_csv('nutri.csv', index=False)
