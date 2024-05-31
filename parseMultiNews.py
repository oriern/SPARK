import unicodedata
from os import listdir, mkdir
from os.path import join, isdir, basename
import os
import io
import re
from nltk import word_tokenize
from unidecode import unidecode
import argparse




def unicode2str_replace(uni):
    string = uni.replace(u"\u2019", "'").replace(u'\u201c', '"').replace(u'\u201d', '"').replace(u'\u2014', '-').\
        replace(u"\u2018", "'").replace(u"\u00b7", "").replace(u'\u2013', '-').replace(u'\u00F1', 'n').replace(u'\u00e8', 'e')\
        .replace(u'\u00e9', 'e').replace(u'\u00a3', 'f').replace(u'\u2022', ' ').replace(u'\u00d7', 'x').replace(u'\u0080', ' ')\
        .replace(u'\u00f9', 'u').replace(u'\u00fa', 'u').replace(u'\u00e1', 'a').replace(u'\u00e0', 'a').replace(u'\u00f2', 'o')\
        .replace(u'\u00f3', 'o').replace(u'\u20ac', 'E').replace(u'\u00fd', 'y').replace(u'\u0107', 'c').replace(u'\u00ed', 'i')\
        .replace(u'\u00ec', 'i')
    string = unidecode(string)
    string = unicodedata.normalize('NFKD', string).encode('ascii').decode("ascii")
    #assert (len(uni)==len(string))
    return string

def MultiNews_handler(database_full_path):
    with io.open(database_full_path, encoding='utf-8') as f:
        doc = f.read()

    doc = unicode2str_replace(doc)
    # new_doc = ''
    # for line in doc.splitlines():
    #     line = line[2:] + '\n'
    #     new_doc += line

    doc = doc.replace(" NEWLINE_CHAR  NEWLINE_CHAR ", "\n")
    doc = doc.replace(" NEWLINE_CHAR NEWLINE_CHAR ", "\n")
    # doc = doc.replace(' |||||', '')


    return doc


def MultiNews_divide_handler(database_full_path):
    with io.open(database_full_path, encoding='utf-8') as f:
        doc = f.read()
    if not os.path.isdir(database_full_path + '_dir'):
        mkdir(database_full_path + '_dir')
    for  topic_idx, topic in enumerate(doc.split('\n')):
        topic = unicode2str_replace(topic)
        topic = topic.replace(" NEWLINE_CHAR  NEWLINE_CHAR ", "\n")
        topic = topic.replace(" NEWLINE_CHAR NEWLINE_CHAR ", "\n")

        if database_full_path.endswith('src'):

            topic_path = join(database_full_path+ '_dir', str(topic_idx + 1))
            mkdir(topic_path)

            for small_doc_idx, small_doc in enumerate(topic.split(' ||||| ')):
                small_doc.replace(' ||||| ','')
                if len(small_doc) < 5:
                    continue

                small_doc_path = join(topic_path,str(small_doc_idx+1) + '.txt' )
                with open(small_doc_path,'w') as f:
                    f.write(small_doc)



        else:
            topic_path = join(database_full_path + '_dir', str(topic_idx + 1) + '.txt')
            with open(topic_path, 'w') as f:
                f.write(topic[2:]) #earase '- '


def parse_data(topic_path):
    for doc_name in listdir(topic_path):
        if doc_name[-11:] == '_parsed.txt':
            continue
        doc_path = join(topic_path,doc_name)

        doc = MultiNews_handler(doc_path)
        with open(doc_path[:-4] + '_parsed.txt','w') as f:
            f.write(doc)


        # MultiNews_divide_handler(doc_path)







# for dataset_dir in [MultiNews_dev_dir, MultiNews_test_dir]:
#     for topic in listdir(dataset_dir):
#
#         topic_path = join(dataset_dir, topic)
#         #print(topic_path)
#         parse_data(topic_path)


parser = argparse.ArgumentParser()

parser.add_argument('-data_path', type=str, default='.')
args = parser.parse_args()



parse_data(args.data_path)
