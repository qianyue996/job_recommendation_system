# -*- coding: utf-8 -*-
import hanlp
# import jieba
import jieba.analyse
import jieba.posseg as pseg
import regex as re

#全角转半角
def full_to_half(sentence):      #输入为一个句子
    change_sentence=""
    for word in sentence:
        inside_code=ord(word)
        if inside_code==12288:    #全角空格直接转换
            inside_code=32
        elif inside_code>=65281 and inside_code<=65374:  #全角字符（除空格）根据关系转化
            inside_code-=65248
        change_sentence+=chr(inside_code)
    return change_sentence

#大写数字转换为小写数字
def big2small_num(sentence):
    numlist = {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","零":"0"}
    for item in numlist:
        sentence = sentence.replace(item, numlist[item])
    return sentence

#大写字母转为小写字母
def upper2lower(sentence):
    new_sentence=sentence.lower()
    return new_sentence

#去除文本中的表情字符（只保留中英文和数字）
def clear_character(sentence):
    pattern1= '\[.*?\]'
    pattern2 = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    pattern3 = '[0-9]'
    line1=re.sub(pattern1,'',sentence)
    line2=re.sub(pattern2,'',line1)
    line3=re.sub(pattern3,'',line2)
    new_sentence=''.join(line3.split()) #去除空白
    return new_sentence

# 结巴中文分词，弃用
# def _jieba(sentence):
#     jieba.load_userdict('data_cleaning/Stopwords/stopwords_full.txt')
#     out = jieba.cut(sentence, cut_all=False)
#     return ' '.join(out)

def process_stopword(sentence):
    stopwords = [line.strip() for line in open('data_cleaning/Stopwords/stopwords_full.txt', 'r', encoding='utf-8').readlines()]
    santi_words = [x for x in sentence if len(x) >1 and x not in stopwords]
    words_split = " ".join(santi_words)
    return words_split

def _hanlp(sentence):
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库
    words = HanLP([sentence])['tok/fine']
    tokens = [word for word in words[0]]
    cleaned_text = ' '.join(tokens).split(' ')
    return cleaned_text

def main(input):
    out = full_to_half(input)
    out = big2small_num(out)
    out = upper2lower(out)
    out = clear_character(out)
    out = _hanlp(out)
    out = process_stopword(out)
    return out

if __name__ == "__main__":
    sequences = '岗位职责：1、主要负责儿童及休闲游戏的开发；2、负责Android产品的移植及发布；3、与策划和美术进行协作，完成设计内容；4、负责游戏后端部分功能开发；任职要求：1、热爱游戏，对游戏开发抱有极大的热情，关注游戏体验；2、计算机相关专业本科毕业；3、熟练掌握Java等开发语言，会C++的优先；4、有好奇心及良好的自我学习能力，具有沟通和协作能力，可以清晰地表达自己的想法；  加分项： 1.会cocos2d-x； 2.会Python及Javascript等脚本语言； 3.有做好的游戏小Demo；'
    output = main(sequences)
    print(output)