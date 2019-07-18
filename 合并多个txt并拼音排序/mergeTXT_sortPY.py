#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-17 14:31:50
# @Author  : Li Shunli (17801063098@163.com)
# @Link    : http://github.com/lishunli-notebook

import os
from pypinyin import lazy_pinyin #利用拼音排序

def sort_pinyin(hanzi_list): #输入list
    """
    将汉字排序。
    """
    hanzi_list_pinyin = []
    hanzi_list_pinyin_alias_dict = {}

    for single_str in list(set(hanzi_list)): #汉字去重
        py_r = lazy_pinyin(single_str) #找到了词的拼音格式：['chuang', 'qian']

        single_str_py = ''.join(py_r) #格式'chuangqian'
        hanzi_list_pinyin.append(single_str_py)

        if single_str_py not in hanzi_list_pinyin_alias_dict.keys(): #如果字典中没有该词
            hanzi_list_pinyin_alias_dict[single_str_py] = [single_str] #床前
        else: #如果字典中已经有该词，则将结果组成一个list
            hanzi_list_pinyin_alias_dict[single_str_py] = list(hanzi_list_pinyin_alias_dict[single_str_py]) + [single_str]
    #到这儿将每个词的拼音和汉字组成了一个字典 {'chuangqian':['床前']} 或 {'ai':['哎', '唉', '爱']}

    hanzi_list_pinyin = list(set(hanzi_list_pinyin)) #拼音去重
    hanzi_list_pinyin.sort() #将拼音排序

    sorted_hanzi_list=[] #汉字按拼音排序后的list
    for single_str_py in hanzi_list_pinyin:

        sorted_hanzi_list += hanzi_list_pinyin_alias_dict[single_str_py] #list + list的格式
        #sorted_hanzi_list.append(hanzi_list_pinyin_alias_dict[single_str_py])
    return sorted_hanzi_list


##将txt合并
def merge_txt(path):
    """
    输入需要排序的txt的路径
    """
     #读取批量txt所在文件夹的路径
    file_names = os.listdir(path) #读取该文件夹下所有的txt的文件名
    file_ob_list = []   #定义一个列表，用来存放刚才读取的txt文件名
    for file_name in file_names:  #循环，以得到它的具体路径
        fileob = path + '/' + file_name #路径mac用/，win用\\
        file_ob_list.append(fileob)

    print(len(file_ob_list)) #输出文件数

    finance = [] #存储所有文件
    print('file name：')
    for name in file_ob_list:
        print(name)
        try:
            finance += [word.strip() for word in open(name, 'r').readlines()] #打开文件，并放入list
        except:
            pass

    return sort_pinyin(finance) #输出按照拼音排序后的list

if __name__ == '__main__':
    path = './NBA'
    finance_str = '\n'.join(merge_txt(path))
    open('NBA_new.txt', 'w', encoding='utf-8').write(finance_str) #输出
