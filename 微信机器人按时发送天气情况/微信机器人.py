#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-18 09:35:08
# @Author  : Li Shunli (17801063098@163.com)
# @Link    : http://github.com/lishunli-notebook

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import itchat
import time
from apscheduler.schedulers.blocking import BlockingScheduler

def getMessage():
  """爬取天气情况，以北京为例"""
    html = urlopen("http://tianqi.sogou.com/beijing/")
    bs0bj = BeautifulSoup(html, features='lxml')
    #具体信息
    total = bs0bj.find("div",{"class":"c-left"}).get_text()
    total_info = [info for info in re.split('\n| |-', total) if info != '']
    #区域
    district = bs0bj.find("div",{"class":"relcity-list"}).get_text()
    district_info = [info for info in re.split('\n', district) if info != '']
    district_dict = {}
    key = 1
    index = 0
    while key:
        try:
            district_dict[district_info[index]] = district_info[index+1]
            index += 2
        except:
            key = 0
    #生活指数
    life = bs0bj.find('div', {'class':'cright-livindex'}).get_text()
    life_info = [word for word in re.split('\n', life) if word != '']
    life_dict = {}
    key = 1
    index = 1
    while key:
        try:
            life_dict[life_info[index]] = [life_info[index+1], life_info[index+2]]
            index += 3
        except:
            key = 0

    message = '今天是：' +  total_info[2] + '年' + total_info[3] + '月' + total_info[4] + '日 ' + total_info[5] + ' ' + total_info[6] + \
    '\n【北京天气】\n\t' + \
    "•[当前温度] %s°C"%(total_info[0]) + \
    '\n\t•' + total_info[1] + ' ' + district_dict['北京'] + \
    '\n\t•'+ total_info[7] + ' ' + total_info[8] + \
    '\n\t•' + total_info[9] + ' ' + total_info[10] + \
    '\n\t•' + total_info[11] + ' ' + total_info[13] + ' ' + total_info[14] +\
    '\n【生活指数】' + \
    '\n\t•穿衣：' + life_dict['穿衣'][1] + \
    '\n\t•雨伞：' + life_dict['雨伞'][1] + \
    '\n\t•紫外线：' + life_dict['紫外线'][1]

    return message  #total_info[0:16], district_dict, life_info

def job():
    message = getMessage()
    my_friend = itchat.search_friends(name=u'备注名')
    friend = my_friend[0]["UserName"]
    itchat.send(msg=message, toUserName=friend)
    print('********发送成功********')

if __name__ == "__main__":
    itchat.auto_login(hotReload=True) #扫码登录
    print('登录成功！')
    sched = BlockingScheduler()
    #sched.add_job(job, 'interval', minutes=10, jitter=10)
    sched.add_job(job, 'cron', hour=7, minute=0) #每天的7点按时发送天气情况
    sched.start()
