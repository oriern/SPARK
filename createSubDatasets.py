import pandas as pd
import numpy as np
import json
import argparse
import os
from os.path import join


def read_generic_file(filepath):
    """ reads any generic text file into
    list containing one line as element
    """
    text = []
    with open(filepath, 'r') as f:
        text = f.read()
        # for line in f.read().splitlines():
        #     text.append(line.strip())
    return text

def read_topic_docs(alignments):
    topic2docs = {}
    for topic in alignments['topic'].drop_duplicates().to_list():
        docs = []
        topic_path = join(args.doc_path, topic)
        for doc in os.listdir(topic_path):
            docs.append(read_generic_file(join(topic_path, doc)))

        topic2docs[topic] = docs

    return topic2docs

def extract_salientSpans(alignments, topic2docs):

    salience_data = {}
    incontext_fusion = {}
    print('number of alignments: ', len(alignments))

    doc_alignments = alignments[['topic', 'documentFile', 'docSentCharIdx',
                                'docSentText', 'docSpanOffsets','docSpanText']].drop_duplicates()

    for topic in doc_alignments['topic'].drop_duplicates().to_list():
        doc_list = topic2docs[topic]
        summary = read_generic_file(join(args.summ_path,topic+'.txt'))

        doc_alignments_topic = doc_alignments[doc_alignments['topic'] == topic]
        spans_list = doc_alignments_topic.apply(lambda x: {'documentFile': x.documentFile, 'docSpanOffsets': x.docSpanOffsets, 'docSpanText': x.docSpanText}, axis=1).to_list()
        salience_data[topic] = {'documents': doc_list, 'salient_spans': spans_list}
        incontext_fusion[topic] = {'documents': doc_list, 'salient_spans': spans_list, 'summary': summary}


    with open(join(args.out_dir_path, 'salience.json'), 'w') as fp:
        json.dump(salience_data, fp)

    with open(join(args.out_dir_path, 'incontext_fusion.json'), 'w') as fp:
        json.dump(incontext_fusion, fp)

    print('number of salient propositions: ',len(doc_alignments.drop_duplicates()))


def extract_clusters(alignments, topic2docs):

    clusters_num  = 0
    evidence_detection_dict = {}
    proposition_clustering_dict = {}
    alignmentsPerClusterList = []
    for topic in alignments['topic'].drop_duplicates().values:
        doc_list = topic2docs[topic]
        df_topic = alignments[alignments['topic'] == topic]
        evidence_detection_dict[topic] = {'docs': doc_list, 'clusters':[]}
        proposition_clustering_dict[topic] = {'input_spans':[], 'clusters': []}
        for index, row in df_topic[['docSpanOffsets', 'docSpanText']].drop_duplicates().iterrows():
            proposition_clustering_dict[topic]['input_spans'].append({'docSpanOffsets': row['docSpanOffsets'],'docSpanText': row['docSpanText']})

        for cluster_idx in df_topic['cluster_idx'].drop_duplicates().values:
            clusters_num += 1
            df_topic_cluster = df_topic[df_topic['cluster_idx'] == cluster_idx]
            alignmentsPerClusterList.append(len(df_topic_cluster))
            query = df_topic_cluster['query'].iloc[0]
            evidence_detection_dict[topic]['clusters'].append({'query':query, 'clusterID': str(cluster_idx), 'spans':[]})
            proposition_clustering_dict[topic]['clusters'].append({'clusterID': str(cluster_idx), 'spans':[]})

            for index, row in df_topic_cluster.iterrows():
                evidence_detection_dict[topic]['clusters'][-1]['spans'].append({'documentFile': str(row['documentFile']),
                                                                           'docSentCharIdx': str(row['docSentCharIdx']),
                                                                           'docSentText': row['docSentText'],
                                                                           'docSpanOffsets': row['docSpanOffsets'],
                                                                           'docSpanText': row['docSpanText']})
                proposition_clustering_dict[topic]['clusters'][-1]['spans'].append(
                    {'documentFile': str(row['documentFile']),
                     'docSentCharIdx': str(row['docSentCharIdx']),
                     'docSentText': row['docSentText'],
                     'docSpanOffsets': row['docSpanOffsets'],
                     'docSpanText': row['docSpanText']})



    print ('clusters number: ', clusters_num)
    print('Num of alignments per cluster: ', np.mean(alignmentsPerClusterList), '(',np.std(alignmentsPerClusterList),')')

    with open(join(args.out_dir_path,"evidence_detection.json"), "w") as outfile:
        json.dump(evidence_detection_dict, outfile)
    with open(join(args.out_dir_path,"proposition_clustering.json"), "w") as outfile:
        json.dump(proposition_clustering_dict, outfile)



def order_clusters(topic_alignments):
    topic_alignments['spanStartIdx'] = topic_alignments['summarySpanOffsets'].apply(lambda x: int(x.split(',')[0]))
    topic_alignments['spanEndIdx'] = topic_alignments['summarySpanOffsets'].apply(lambda x: int(x.split(',')[-1]))
    topic_alignments['spanStartIdx_cluster'] = topic_alignments.groupby('cluster_idx')['spanStartIdx'].transform(min)
    topic_alignments['spanEndIdx_cluster'] = topic_alignments.groupby('cluster_idx')['spanEndIdx'].transform(max)
    ordered_topic_alignments = topic_alignments.sort_values(by=['spanStartIdx_cluster', 'spanEndIdx_cluster'])
    # ordered_topic_alignments['order_idx'] = topic_alignments.groupby('cluster_idx').ngroup()


    return ordered_topic_alignments

def extract_textPlanning(alignments):


    planning_dict = {}

    for topic in alignments['topic'].drop_duplicates().values:

        df_topic = alignments[alignments['topic'] == topic]
        ordered_topic_alignments  = order_clusters(df_topic)
        ordered_topic_alignments['sent_group_idx'] = ordered_topic_alignments.groupby('scuSentCharIdx').ngroup()
        planning_dict[topic] = {'clusters':[]}
        cluster_order_idx = 0
        for cluster_idx in ordered_topic_alignments['cluster_idx'].drop_duplicates().values:
            df_topic_cluster = ordered_topic_alignments[ordered_topic_alignments['cluster_idx'] == cluster_idx]
            planning_dict[topic]['clusters'].append({'clusterID': str(cluster_idx), 'spans':[],
                                                     'order_idx': cluster_order_idx,
                                                     'sent_group_idx': str(df_topic_cluster['sent_group_idx'].iloc[0])})

            cluster_order_idx += 1
            for index, row in df_topic_cluster.iterrows():
                planning_dict[topic]['clusters'][-1]['spans'].append({'documentFile': str(row['documentFile']),
                                                                           'docSentCharIdx': str(row['docSentCharIdx']),
                                                                           'docSentText': row['docSentText'],
                                                                           'docSpanOffsets': row['docSpanOffsets'],
                                                                           'docSpanText': row['docSpanText']})




    with open(join(args.out_dir_path,"planning.json"), "w") as outfile:
        json.dump(planning_dict, outfile)


def sentenceFusion(alignments):

    fusion_dict = {}

    for topic in alignments['topic'].drop_duplicates().values:
        df_topic = alignments[alignments['topic'] == topic]
        fusion_dict[topic] = []

        for scuSentence in df_topic['scuSentence'].drop_duplicates().values:
            df_topic_sent = df_topic[df_topic['scuSentence'] == scuSentence]
            fusion_dict[topic].append({'fused_sentence': scuSentence, 'clusters': []})

            for cluster_idx in df_topic_sent['cluster_idx'].drop_duplicates().values:

                df_topic_sent_cluster = df_topic_sent[df_topic_sent['cluster_idx'] == cluster_idx]
                fusion_dict[topic][-1]['clusters'].append({'clusterID': str(cluster_idx), 'spans': []})

                for index, row in df_topic_sent_cluster.iterrows():
                    fusion_dict[topic][-1]['clusters'][-1]['spans'].append({'documentFile': str(row['documentFile']),
                                                                          'docSentCharIdx': str(row['docSentCharIdx']),
                                                                          'docSentText': row['docSentText'],
                                                                          'docSpanOffsets': row['docSpanOffsets'],
                                                                          'docSpanText': row['docSpanText']})

    with open(join(args.out_dir_path, "fusion.json"), "w") as outfile:
        json.dump(fusion_dict, outfile)




parser = argparse.ArgumentParser()
parser.add_argument('-alignments_path', type=str,  default='/home/nlp/ernstor1/alignmentDerivedDatasets/evidence_detection/data_alignments_gold_w_query_fixed.csv')
parser.add_argument('-doc_path', type=str,  default='/home/nlp/ernstor1/alignmentDerivedDatasets/data/MultiNews/test')
parser.add_argument('-summ_path', type=str,  default='/home/nlp/ernstor1/alignmentDerivedDatasets/data/MultiNews/test/summaries')
parser.add_argument('-out_dir_path', type=str, default='publish_data/')
args = parser.parse_args()

if __name__ == "__main__":


    alignments = pd.read_csv(args.alignments_path)
    topic2docs = read_topic_docs(alignments)
    extract_salientSpans(alignments, topic2docs)
    extract_clusters(alignments, topic2docs)
    extract_textPlanning(alignments)
    sentenceFusion(alignments)