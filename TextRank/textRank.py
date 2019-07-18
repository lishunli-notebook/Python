#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-16 10:39:40
# @Author  : Li Shunli (17801063098@163.com)
# @Link    : http://github.com/lishunli-notebook
# @Method  : 将内容看成句子的组合，构成一个无向图，每个句子为顶点，句子之间的相似度为边。初始化得分，通过公式计算，直到得分收敛。

import re
import numpy as np

class textRank(object):
    """textRank"""

    def __init__(self, userDict=None):
        self.userDict = userDict

    def sim(self, content=None, min_count=2):
        """使用word2vec计算相似度"""

        import word2vec
        import jieba
        from sklearn.metrics.pairwise import cosine_similarity
        from zhon.hanzi import non_stops #不包括分句的标点

        punc = non_stops + ' ' + '@' + '(' + ')'#标点符号


        if self.userDict != None:
            jieba.load_userdict(self.userDict)

        copurs = ' '.join([word for word in jieba.lcut(content) if word not in punc]) #wordvec的语料库
        open('content_textRank.txt', 'w+', encoding='utf-8').write(copurs) #保存语料库
        word2vec.word2vec('content_textRank.txt', 'content_textRank.bin', min_count=min_count) #保存模型
        model = word2vec.load('content_textRank.bin') #加载模型

        self.sentences = re.split(r'。|？|！', content) #这是内容按句子分开后的结果
        seg_corpus = [[word for word in jieba.lcut(sentence) if word not in punc] for sentence in self.sentences if sentence not in  ['', ' '] ]

        sentence_vector = []
        for item in seg_corpus:
            vector = np.zeros(model.vectors.shape[1])
            k = 0
            for word in item:
                if word in model.vocab:
                    vector += model.get_vector(word)
                    k += 1
            if k > 0:
                sentence_vector.append(vector / k)
            # else:
            #     sentence_vector.append(vector) #可能有零向量

        self.similarity = cosine_similarity(np.array(sentence_vector))

        return self.similarity #这就是句子与句子之间的权重


    def calculate_score(self, scores, i):
        """
        计算第i个句子的得分。
        """

        n = len(self.similarity)
        d = 0.85 #阻尼系数
        added_score = 0.0 #第i'个句子新的得分

        for j in range(n):
            fraction = 0.0 #分子
            denominator = 0.0 #分母
            # 先计算分子
            if i != j:
                fraction = self.similarity[i, j] * scores[j]
            # 计算分母
                for k in range(n):
                    if k != j:
                        denominator += self.similarity[j, k]
                added_score += fraction / denominator
        # 算出最终的分数
        return (1 - d) + d * added_score


    def getSummary(self, content=None, min_count=2, threshold=1e-5, proportion=0.25):
        """
        初始化得分，每个句子的得分均设为0.5。通过迭代后输出句子.
        params:
            proportion:摘要比例，若在(0, 1]，则为比例；若为>1，则必须为整数，输出摘要的句子数目。
        """
        self.sim(content=content, min_count=min_count)

        score_new = [0.5 for i in range(len(self.similarity))]
        score_old = [0.0 for i in range(len(self.similarity))]

        while abs(np.array(score_new) - np.array(score_old)).min() > threshold: #停止条件

            score_old = score_new.copy()

            for i in range(len(self.similarity)):
                score_new[i] = self.calculate_score(scores=score_old, i=i)

        if proportion <= 1:
            n = int(proportion * len(self.sentences))#摘要的比例大小
        else:
            n = proportion #摘要的句子数

        summary = []
        for index in np.array(score_new).argsort()[:-(n+1):-1]: #分数最大的n个
            summary.append(self.sentences[index])

        return summary





if __name__ == '__main__':

    content = '伴随着世界杯的落幕，俱乐部联赛筹备工作又成为主流，转会市场必然也会在世界杯的带动下风起云涌，\
    不过对于在本届赛事上大放异彩的姆巴佩而言，大巴黎可以吃一颗定心丸，世界杯最佳新秀已经亲自表态：留在巴黎哪里也不去。\
    在接受外媒采访时，姆巴佩表达了继续为巴黎效忠的决心。“我会留在巴黎，和他们一起继续我的路途，我的职业生涯不过刚刚开始”，\
    姆巴佩说道。事实上，在巴黎这座俱乐部，充满了内部的你争我夺。上赛季，卡瓦尼和内马尔因为点球事件引发轩然大波，\
    而内马尔联合阿尔维斯给姆巴佩起“忍者神龟”的绰号也让法国金童十分不爽，为此，姆巴佩的母亲还站出来替儿子解围。\
    而早在二月份，一场与图卢兹的比赛，内马尔也因为传球问题赛后和姆巴佩产生口角。由此可见，巴黎内部虽然大牌云集，\
    但是气氛并不和睦。内马尔离开球队的心思早就由来已久，而姆巴佩也常常与其它俱乐部联系在一起，在躲避过欧足联财政公平法案之后\
    ，巴黎正在为全力留下二人而不遗余力。好在姆巴佩已经下定决心，这对巴黎高层而言，也算是任务完成了一半。本届世界杯上，\
    姆巴佩星光熠熠，长江后浪推前浪，大有将C罗、梅西压在脚下的趋势，他两次追赶贝利，一次是在1/8决赛完成梅开二度，\
    另一次是在世界杯决赛中完成锁定胜局的一球，成为不满20岁球员的第二人。另外他在本届赛事中打进了4粒入球，\
    和格列兹曼并列全队第一。而对巴黎而言，他们成功的标准只有一条：欧冠。而留下姆巴佩，可以说在争夺冠军的路上有了仰仗，\
    卡瓦尼在本届世界杯同样表现不错，内马尔虽然内心波澜，但是之前皇马官方已经辟谣没有追求巴西天王，三人留守再度重来，\
    剩下的就是图赫尔的技术战术与更衣室的威望，对图赫尔而言，战术板固然重要，但是德尚已经为他提供了更加成功的范本，\
    像团结法国队一样去团结巴黎圣日耳曼，或许这才是巴黎取胜的钥匙。'
    a = textRank(userDict='user_dict.txt')
    print(a.getSummary(content, proportion=2))













