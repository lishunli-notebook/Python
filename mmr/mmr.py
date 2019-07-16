#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-07-10 16:42:10
# @Author  : Li Shunli (17801063098@163.com)
# @Link    : http://github.com/lishunli-notebook
# @Method : 基于词频的mmr/似乎可以用word2vec进行

import pandas as pd
import os
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class mmr(object):

    def __init__(self, stopPath=None, userDict=None, doc=None, title=None):
        """
        params:
            stopPath:停用词地址
            doc:文档取摘要。
            title:文档标题。
        """
        self.stopPath = stopPath #a string
        self.userDict = userDict
        self.doc = doc #a string
        self.title = title

    def stopWords(self):
        """
        停用词。
        """
        stopwords=[word.strip() for word in open(self.stopPath, 'r', encoding='utf-8').readlines()]
        return stopwords

    def segData(self, name):
        """
        分词
        """
        if self.userDict != None:
            jieba.load_userdict(self.userDict)

        seg_list = [word for word in jieba.lcut(name) if word not in self.stopWords()]
        return " ".join(seg_list)

    def cleanDoc(self):
        """
        清理原文档

        """
        self.sentences = []   #原句子
        self.clean = []   #干净有重复的句子
        self.sentence_dict = {}   #字典

        parts = re.split(r'。|？|！', self.doc)#句子拆分
        for part in parts:
            if part == '':
                continue
            cl = self.segData(part)#句子切分以及去掉停止词
            if cl == '':
                continue
            self.sentences.append(part) #原本的句子
            self.clean.append(cl) #干净有重复的句子
            self.sentence_dict[cl] = part #字典格式

        self.setClean = set(self.clean) #干净无重复的句子

    def calculateSimilarity(self, sentence, doc):#根据句子和句子，句子和文档的余弦相似度
        """
        计算句子文档的相似度。基于词频。
        params:
            sentence:句子
            doc:文档
        return:
            句子和文档的相似度
        """
        if len(doc) == 0:
            return 0

        vocab = {} #所有词的词典，用于初始化CountVectorizer
        for word in sentence.split():
            vocab[word] = 0

        doc_sen = '' #文档的句子
        for sen in doc:
            doc_sen += (sen + ' ')#所有剩余句子合并
            for word in sen.split():
                vocab[word] = 0

        cv = CountVectorizer(vocabulary=vocab.keys()) #所有词初始化
        docVector = cv.fit_transform([doc_sen])  #文档的所有句子
        sentenceVector = cv.fit_transform([sentence])  #某一个句子
        return cosine_similarity(docVector, sentenceVector)[0][0] #计算相似度


    def getSummary(self, proportion=0.25, alpha=0.7, outPut='seg', C=2):
        """
        得到文章的摘要。
        params:
            proportion:摘要比例，若在(0, 1]，则为比例；若为>1，则必须为整数，输出摘要的句子数目。
            alpha:准确度与多样性系数，取值范围[0, 1]， 值越大，准确度越高；值越小，多样性越大。
            outPut:seg or raw，输出分词后或分词前的摘要， 默认分词后seg。
            C:标题的正则项系数。C越大，表明句子和标题的相似度越强。
        return:
            outPut='seg',输出摘要的分词结果。
            outPut='raw',输出摘要的原文。
        """
        self.cleanDoc()
        scores = {}
        for data in self.setClean:
            temp_doc = self.setClean - set([data])#在除了当前句子的剩余所有句子
            score = self.calculateSimilarity(data, list(temp_doc)) #计算当前句子与剩余所有句子的相似度
            scores[data] = score #得到句子与全文的相似度列表

        ##计算mmr
        if proportion <= 1:
            n = proportion * len(self.sentences)#摘要的比例大小
        else:
            n = proportion #摘要的句子数

        self.summarySet = []
        while n > 0:
            mmr = {}
            for sentence in scores.keys():
                if not sentence in self.summarySet:
                    #mmr公式
                    if self.title == None:
                        mmr[sentence] = alpha * scores[sentence] - (1-alpha) * self.calculateSimilarity(sentence, self.summarySet)

                    else:
                        mmr[sentence] = alpha * scores[sentence] \
                        - (1-alpha) * self.calculateSimilarity(sentence, self.summarySet) \
                        + C * self.calculateSimilarity(sentence, [self.segData(self.title)])

            selected = max(mmr.items(), key=lambda x:x[1])[0] #最大的一个
            self.summarySet.append(selected)
            n -= 1

        self.raw = []  #原文
        for sen in list(self.summarySet):
            self.raw.append(self.sentence_dict[sen])

        if outPut == 'seg':
            return self.summarySet  #输出分词后的摘要
        elif outPut == 'raw':
            return self.raw
        else:
            print('outPut Error!')

    def printSentence(self, col='red'):
        """
        打印摘要句子在原文的位置，用col标出。
        params:
            col:摘要句颜色，默认红色。
        """

        from termcolor import colored

        for sentence in self.clean:
            if sentence in self.summarySet:
                print (colored(self.sentence_dict[sentence].lstrip(' '), col))
            else:
                print (self.sentence_dict[sentence].lstrip(' '))

if __name__ == '__main__':
    doc = '对于很多小白来说，相比于直接参与股票投资，选择投资基金，由基金经理为你精选个股，合理规划资产配置更能降低踩雷的风险。\
    我们都知道基金按照投资方式可以分为：一次性投资和定投，它们之间到底有什么区别？到底哪个更靠谱呢？老巴今天给大家梳理一下。 \
    一、 一次性投资和定投的区别顾名思义，一次性投资就是一次性将资金全部投入基金的行为，定投就是定期定额投资基金的意思。\
    那两者有什么区别呢？1、 投资目的不同一次性投资一般是为了短期获取超额利润而做出的投资行为，有一定的投机性。\
    说白了就是看好短期能有一个大反弹或者是大牛市，是一种进攻性投资。定投不是为了短期收益，而是一种长期的理财投资，是一种防守型投资。\
    2、 资金来源一次性投资的资金是积蓄，是长时间的资金累积。定投的资金是每月的闲置零钱，可以是几百块，也可以是几千，定投能让这部分钱“活“起来。\
    3、 投资能力一次性投资对投资者的要求很高，首先能承受较高的风险，因为市场的走势往往不如你所愿。另外需要对时间点把握非常准确，\
    需要一定的投资经验，才能在合适的时间低买高卖。定投适合小白投资者，因为随时都可以进场，尤其是熊市里进场效果更好，用时间获取收益，\
    到牛市里分批止盈即可。4、 情绪控制一次性投资考验心态，由于是一次性的，金额一般较大，对于下跌一般人往往扛不住，如果短期没有达到预想的收益，\
    反而被套住，不止损就意味着要长期抗住，这反而失去了一次性投资的意义，但是止损又不甘心，纠结起来既劳心又伤财。定投就轻松多了，甚至不需要每天看盘\
    ，只需要在下跌的时候沉住气定投，上涨的时间段分批落袋为安即可。二、 不同市场两者的表现1、 牛市里一般来说，单边上行市场，定投跑的没有一次性投入快。\
    因为定投是逐笔投入，不像一次性投资可以让手里资金享受到整个牛市。老巴取了今年一季度小牛市的数据，以华宝沪深300增强（003876）为例，\
    每周定投一千元，13期定投和一次性定投数据如下：数据来源：Wind，定投品种：华宝沪深300增强A（003876），实际情况请以真实交易为准 ，下同。\
    结论：牛市里定投收益不如一次性投资收益，上涨的波段可以适当减少定投的金额，甚至可以分批落袋为安。2、 熊市里老巴取了去年全年熊市的数据，\
    每月定投一千元，12期定投和一次性定投数据如下：结论：熊市里定投抗跌性强于一次性投资，在下跌的时候，可以适当增加定投的金额，为牛市蓄力做准备。\
    3、 熊转牛老巴取了去年下半年熊市到今年上半年小牛市的数据，每月定投一千元，12期定投和一次性定投数据如下：\
    结论：熊转牛，这是定投发挥最出色的一个阶段。由于在下跌阶段积攒了较多筹码，牛市到来的时候会赚取更高的收益，熊市跌的越多，\
    牛市的时候赚的越多，远超一次性投资。三、总结总的来说，定投适合小白投资者，一次性投资适合高手。而且A股常态是牛短熊长，在熊市和熊转牛的市场，\
    定投更能发挥威力。比如当下震荡市场，沪指围绕3000点上下波动，正是播种撒苗的好时候~@今日话题 @蛋卷基金 $新城控股(SH601155)$ \
    $中国平安(SH601318)$ $贵州茅台(SH600519)$ '
    title = '一次性投资和定期定额投资有什么区别？'
    
    a = mmr('中文停用词.txt', '财经词典.txt', doc, title)
    b = a.getSummary(outPut='seg')
    for x in b:
        print('---'*8)
        print(x)

    print('-*-*-*-*-*'*8)
    a2 = mmr('中文停用词.txt', '财经词典.txt', doc)
    b2 = a2.getSummary(outPut='raw')
    for x in b2:
        print('---'*8)
        print(x)





