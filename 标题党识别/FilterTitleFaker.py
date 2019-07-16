#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-11 13:29:27
# @Author  : Li Shunli (17801063098@163.com)
# @Link    : http://github.com/lishunli-notebook
# @Method : 将所有数据分词（注意不要去除停用词）后

import os
import pandas as pd
import numpy as np
import mmr  #主题句抽取

class Filter_title_faker(object):

    def __init__(self, title, content,  stopPath=None, user_dict=None):
        self.title = title
        self.content = content
        self.stopPath = stopPath
        self.user_dict = user_dict

    def Stopwords(self):
        """
        符号、文字停用词.
        return:
            punc:标点
            stopwords:停用词
        """
        from zhon.hanzi import non_stops #不包括分句的标点

        self.punc = non_stops + ' ' + '@' + '(' + ')'#标点符号

        if self.stopPath == None:
            self.stopwords = []
            print('stopPath is None')
        else:
            self.stopwords = [word.strip() for word in open(self.stopPath, 'r', encoding='utf-8').readlines()]
        self.stopwords.append(' ') #停用词

        return self.punc, self.stopwords

    def seg(self, df):
        """
        分词去标点。
        """
        from lxml.html.clean import Cleaner
        import re
        import jieba

        if self.user_dict != None:
            jieba.load_userdict(self.user_dict)
        result_seg = []
        c = Cleaner()
        for i in df:
            try:
                result0 = c.clean_html(i)
                result1 = re.sub(r'<.*?>', ' ', result0) #清理数据
            except:
                result1 = i

            result2 = ' '.join([word for word in jieba.lcut(result1) if word not in self.punc]) #将结果分词并去标点，但保留。？！这三个分句的符号

            result_seg.append(result2)

        return result_seg #输出类型[' ', ' ']， 每个引号为内容的分词结果

    def Segment(self):
        """
        处理数据并分词。
        """
        self.title_seg = self.seg(self.title)
        self.content_seg = self.seg(self.content)

        return self.title_seg, self.content_seg

    def Plot_wordcloud(self, key='content',
        font_path='/System/Library/Fonts/PingFang.ttc',
        background_color='black', mask=None, **kwgs):
        """
        画词云图
        **kwgs: save_path=None
        """
        import wordcloud
        import matplotlib.pyplot as plt
        import re

        if key == 'content':
            df_wc = self.content_seg
        else:
            df_wc = self.title_seg
        df = [word for word in re.split(r' ', ' '.join(df_wc)) if word not in self.stopwords]
        wc = wordcloud.WordCloud(
                font_path = font_path, #设置字体格式
                width = 500,  #设置画布宽度
                height = 500, #设置画布高度
                background_color = background_color, #设置背景颜色
                mask = np.array(mask) if mask != None else None) #设定背景形状
        wc.generate(' '.join(df))
        plt.imshow(wc)    #画出词云图
        plt.axis('off')   #关闭坐标轴

        try:
            plt.savefig(kwgs['save_path'], dpi=300, bbox_inches='tight')
        except:
            pass

        plt.show()
        plt.close()


    def Get_tfidf(self):
        """
        计算内容的tfidf。
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        tf = TfidfVectorizer()
        result = tf.fit_transform(self.content_seg)
        self.words = tf.get_feature_names()
        self.weight = result.toarray()

        return self.words, self.weight

    def Train_word2vec(self, model_name='content_seg', size=100, window=5):
        """
        训练模型，获得词向量。（内容和标题，该内容仅去了不必要的标点，并未去掉停用词）
        #需要注意，对于一个预测类的内容和标题，需要重新训练模型。或者要求输入的title和content已经包含了所有数据。
        """
        import word2vec

        total_corpus = self.content_seg + self.title_seg #两个列表相加
        #df_content_word2vec =   ###[' '.join(x) for x in total_corpus]
        file_path = os.getcwd() + '/' + model_name + '.txt'
        open(file_path, 'w+', encoding='utf-8').write(' '.join(total_corpus)) #保存数据

        self.output_path = os.getcwd() + '/' + model_name + '.bin'
        word2vec.word2vec(file_path, self.output_path, size=size, window=window)

        return self.output_path #导出模型保存的文件

    def K_means_cluster(self, n_clusters=5):
        """
        K-means clustering.
        params:
            n_clusters:  number of clusters.
        return:
            Vectors of each central.
        """
        from sklearn.cluster import KMeans
        import word2vec
        import re

        ##找到各个title的词向量
        model = word2vec.load(self.output_path)

        title_vector_list = []
        for i in self.title_seg:
            vector = np.zeros(model.vectors.shape[1])
            item = [word for word in re.split(r' ', i) if word not in self.stopwords] #每个标题的词语
            if len(item) > 0:
                k = 0 #做平均
                for word in item:
                    if word in model.vocab:#可能有的词不再词向量中
                        vector += model.get_vector(word)
                        k += 1

                if k != 0: #若标题中没有任何一个关键词，则不添加到title_vector_list
                     title_vector_list.append(vector / k)

        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(np.array(title_vector_list))

        # center = [] #聚类中心单位化
        # for x in list(kmeans.cluster_centers_):
        #     center.append(x/((x * x).sum()) ** (.5))

        self.cluster_center = kmeans.cluster_centers_ #返回中心
        return self.cluster_center

    def Get_abstract(self, oneContent, oneTitle=None, outPut='seg'):
        """
        利用mmr包.
        params:
            oneContent:内容
            oneTitle:标题
            outPut:输出内容，分词结果/原文
        return:
            输出摘要
        """

        MMR = mmr.mmr(stopPath=self.stopPath, doc=oneContent, title=oneTitle)
        abstract = MMR.getSummary(outPut=outPut)
        return ' '.join(abstract) #是一个字符串，并且用空格分开

    def Get_content_vector(self):
        """
        获得每个内容的词向量
        """
        import word2vec
        import re

        self.vector_result = []
        #distance = []
        model = word2vec.load(self.output_path)

        content_vector = [] #所有文档的加权平均向量
        for i in range(len(self.content_seg)):
            vector = np.zeros(model.vectors.shape[1])
            content_tfidf = 0

            #content = [word for word in re.split(r' ', self.content_seg[i]) if word not in self.stopwords] #所有内容词语
            content = [word for word in re.split(r' ', self.Get_abstract(self.content[i]))] #主题句词语
            for word in content:
                try:
                    word_vector = model.get_vector(word)
                    word_tfidf = self.weight[i, self.words.index(word)]
                    content_tfidf += word_tfidf
                    vector += word_tfidf * word_vector
                except:
                    pass

            if content_tfidf == 0: #content中没有词
                content_vector.append(vector + 1e-5) #保证该向量不为零向量，否者计算相似度将出错
            else:
                content_vector.append(vector/content_tfidf)  #做加权平均

        self.content_vector = np.array(content_vector)

        return self.content_vector

    def Get_similarity(self):
        """
        利用array格式计算相似度。
        """
        ###处理content中可能出现为0向量的向量：将其处理为全体向量的均值


        try:
            cent = self.cluster_center / np.linalg.norm(self.cluster_center, axis=1, keepdims=True) #归一化
            cont = self.content_vector / np.linalg.norm(self.content_vector, axis=1, keepdims=True) 
            self.similarity = cont.dot(cent.T).max(axis = 1)
            return self.similarity
        except:
            print('There is only one piece of data！')

    def labels_(self, threshold=0.2):
        """
        给定阈值，输出标签。
        1, title faker：similarity < threshold
        0: normal
        """
        label = self.similarity.copy()
        label[label < threshold] = 1
        label[label >= threshold] = 0

        return label


    def main(self, threshold=0.2, n_clusters=5, plot_wordcloud=True):
        """
        主函数：可以直接获得标签值。
        params:
            threshold:判定为标题党的阈值，默认0.2
            stopPath:停用词路径
            user_dict:个人用户词典
            n_clusters:标题聚类数目
            plot_wordcloud:是否画出词云图，默认True
        return:
            labels:每个文章的标签，1为标题党。
        """
        self.Stopwords()
        self.Segment()
        if plot_wordcloud:
            self.Plot_wordcloud()
        self.Get_tfidf()
        self.Train_word2vec()
        self.K_means_cluster(n_clusters=n_clusters)
        self.Get_content_vector()
        self.Get_similarity()

        return self.labels_(threshold=threshold)

if __name__ == '__main__':
    data = pd.read_csv('财经新闻.csv')
    df = data[:100]

    test = Filter_title_faker(df.TITLE, df.CONTENT, stopPath='中文停用词.txt', user_dict='财经词典.txt')
    a = test.main(plot_wordcloud=False)
    #print('lable:', a)
    #print('K center', test.K_means_cluster())
    #print('similarity', test.Get_similarity())
    print('----'*10)
    print('bad:', (a == 1).sum(), '\t', 'good:', (a == 0).sum())



