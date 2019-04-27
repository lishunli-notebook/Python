#! usr/bin/env python3
# -*- coding: utf-8 -*-
# Author:Lishunli
# E-mail:969984602@qq.com

'''
This is a simple Web Scraping tool of getting news.

'''


import requests                        #打开链接
from bs4 import BeautifulSoup          #解码链接
from collections import OrderedDict    #特定字典？
from urllib.parse import urlencode     #输入内容转码
import time                            #获取时间
import pandas as pd
import numpy as np


def sina_search(page=2):
    """Function to get  web list pages for a given keyword and page number.

    Args:
        keywords: manual input.
        page: The page number, default 2.

    Returns:
        newsData: A dataframe of the contents in sina web about keywords.
        
    """

    import time as TM
    
    ##输出如期信息
    print(TM.strftime('现在是北京时间：%Y-%m-%d %A %H:%M:%S',TM.localtime(TM.time())))
    
    ###手动输入需要查询的内容
    key_words = input('请输入新闻关键词（空格分开，回车结束）：') 
    key_word = re.split(r'\s', key_words) #按照空格分开
    
    #初始化时间
    time_begin = TM.time()
    
    k = 0  #初始化数据框的index
    newsData = pd.DataFrame(columns = ['title', 'date','time', 'sourse', 'abstract','url', 'content'])
    
    ###第一层循环：针对输入的词
    for compRawStr in key_word:
        time_keyword = TM.time()
        # Dealing with character encoding
        comp = compRawStr.encode('gbk')  #解码名字
        d = {'q': comp}
        pname = urlencode(d)  #名称编码
        
        count_info = 0  #记录爬取数据条数
        ###第二层循环：针对每一个词的页数进行循环
        for i in range(1, page+1):  #从第一页开始
            href = 'http://search.sina.com.cn/?%s&range=all&c=news&sort=time&col=&source=&from=&country=&size=&time=&a=&page=%s'%(pname, i) # comp -> first %s; page -> 2nd %s; col=1_7 -> financial news in sina
            html = requests.get(href) #打开链接
            # Parsing html
            soup = BeautifulSoup(html.content, 'html.parser')  #解码链接
            divs = soup.findAll('div', {"class": "box-result clearfix"})  #解码后找到相应内容
            
            ###第三个循环：对每一页的内容进行循环
            for div in divs:  #找到各个文件下的东西
                head = div.findAll('h2')[0] #标题和链接
                # News title
                titleinfo = head.find('a')   #进入子目录
                title = titleinfo.get_text() #获得a的内容：即标题
                # News url
                url = titleinfo['href']  #获得a标题链接
                # Other info
                
                otherinfo = head.find('span', {"class": "fgray_time"}).get_text()  #其他信息
                source, date, time = otherinfo.split()   #空格分隔
               
                # News abstract
                abstract = div.find('p', {"class": "content"}).get_text()   #摘要文本
                
                ##找到链接中的具体内容
                content_htlm = requests.get(url)  #打开标题链接
                content_soup = BeautifulSoup(content_htlm.content, 'html.parser')  #解码链接
                content = content_soup.find('div', {'class': 'article'}).get_text()  #获得内容
                
                newsData.loc[k,:] = [title, date, time, source, abstract, url, content]    #将数据放在一个数据框中
                
                k += 1 #index更新
                count_info += 1
        print('用时：%.4fs，'%(TM.time() - time_keyword), '找到有关【%s】数据%s页，共%s条。'%(compRawStr, page, count_info))
    print('-------'*8, '\n用时：%.4fs共爬取%s条数据。'%(TM.time() - time_begin, k))   
    return newsData


#找到新浪新闻前5页内容
news_about_finance = sina_search(5)  


#详细数据
news_about_finance.head(20)














