"""
This is to get a txt list of picture name
"""

import os


def search_file(start_dir, target):
    os.chdir(start_dir)
    #count = 0
#//Users/wutong/Downloads/train_set/clips/0313-1/8580/
    for each_file in os.listdir(os.curdir):
        #if count == 820:
            #break

        ext = os.path.splitext(each_file)[1]
        if ext in target and '20' in each_file:
            vedio_list.append('/' + os.getcwd() + os.sep + each_file + os.linesep)  # 使用os.sep更标准
        if os.path.isdir(each_file):
            search_file(each_file, target)  # recursive
            os.chdir(os.pardir) 
        #count += 1


start_dir = '//Users/wutong/Downloads/train_set/clips/0313-1/'
program_dir = os.getcwd()

target = ['.jpg']
vedio_list = []

search_file(start_dir, target)

f = open('//Users/wutong/Desktop/prob_map/list_test_copy.txt', 'w')
f.writelines(vedio_list)
f.close()
