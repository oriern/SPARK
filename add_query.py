import pandas as pd
import argparse
import os
from nltk import word_tokenize
from itertools import chain
import numpy as np
from nltk import sent_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('-prediction_path', type=str, default='/home/nlp/ernstor1/alignmentEval/annotation_checker_pkg/Task3_Datasets/devMultiNews_test0_checkpoint-2000_negative.csv')#'/home/nlp/ernstor1/alignmentDerivedDatasets/highlighting/models/MultiNews_only/test_results_None.csv')
# parser.add_argument('-prediction_metadata_path', type=str, default='/home/nlp/ernstor1/alignmentDerivedDatasets/highlighting/data/CDLM/MultiNews_test_CDLM_allAlignments_fixed_truncated_metadata.csv')#'/home/nlp/ernstor1/alignmentDerivedDatasets/highlighting/data/CDLM/MultiNews_test_CDLM_allAlignments_fixed_truncated_metadata.csv')
# parser.add_argument('-prediction_path_w_metadata', type=str, default='/home/nlp/ernstor1/alignmentDerivedDatasets/chatGPT/predicted_highlights_gpt35.csv')
parser.add_argument('-gold_path', type=str, default='/home/nlp/ernstor1/alignmentDerivedDatasets/test_original_text.csv')#'/home/nlp/ernstor1/alignmentDerivedDatasets/highlighting/models/MultiNews_only/test_results_None.csv')
parser.add_argument('-summaries_path', type=str, default='/home/nlp/ernstor1/autoAlignment/eval_data/summaries/MultiNews/test/')
args = parser.parse_args()


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

def read_predicted(gold_selected_topics):
    alignments = pd.read_csv(args.prediction_path)
    # selected_topics = alignments['topic'].drop_duplicates().to_list()[:100]
    #
    #
    alignments = alignments[alignments['topic'].isin(gold_selected_topics)]  #use only the topics from gold
    positive_alignments = alignments[alignments['pred_prob'] >= 0.5]


    return positive_alignments


def read_gold():
    # predictions = pd.read_csv(args.gold_path)

    metadata = pd.read_csv(args.gold_path)#pd.read_csv('/home/nlp/ernstor1/alignmentDerivedDatasets/full_test.csv')
        # '/home/nlp/ernstor1/alignmentDerivedDatasets/highlighting/data/CDLM/MultiNews_test_CDLM_allAlignments_fixed_truncated_metadata.csv')

    # assert (len(predictions) == len(metadata))
    # metadata.insert(2, "prediction", predictions['prediction'])

    metadata['prediction'] = 1

    metadata['topic_id'] = metadata['topic'].apply(lambda x: int(x.split('est')[1]))

    # metadata = metadata[metadata['topic_id'] < 4]

    # predictions = metadata
    metadata = metadata.dropna()
    return metadata




def intersection_over_union(span1, span2):
    # Function to calculate the overlap percentage between two spans
    intersection = len(span1.intersection(span2))
    union = len(span1.union(span2))
    return intersection / union if union > 0 else 0

def alignments2clusters(alignments, overlap_threshold=0.5):
    # Create a new column 'cluster_idx' and initialize with -1
    alignments['cluster_idx'] = -1



    # Iterate through each unique topic in the DataFrame
    for topic in alignments['topic'].unique():
        # Create a list to store the clusters for the current topic
        clusters = []

        # Filter the DataFrame by the current topic
        alignments_topic = alignments[alignments['topic'] == topic]

        # Iterate through each row in the filtered DataFrame
        current_cluster_idx = 0
        for index, row in alignments_topic.iterrows():
            # span_start, span_end = map(int, row['summarySpanOffsets'].replace(' ', '').split(';')[0].split(','))
            offset = offset_str2list(row['summarySpanOffsets'])
            # offset = offset_decreaseSentOffset(sentOffset, offset)
            ranges = [range(marking[0], marking[1]) for marking in offset]
            ranges = set(chain(*ranges))
            # span = set(range(span_start, span_end + 1))

            # Check if the span overlaps with any existing cluster with at least 60% overlap
            merged = False
            for cluster_idx, cluster in enumerate(clusters):
                overlap_percentage = intersection_over_union(ranges, cluster)
                if overlap_percentage >= overlap_threshold:
                    alignments.at[index, 'cluster_idx'] = cluster_idx
                    cluster.update(ranges)
                    merged = True
                    break

            # If the span does not overlap with any existing cluster, create a new cluster
            if not merged:
                alignments.at[index, 'cluster_idx'] = current_cluster_idx
                clusters.append(ranges)
                current_cluster_idx += 1

    return alignments
def offset_str2list(offset):
    return [[int(start_end) for start_end in offset.split(',')] for offset in offset.split(';')]

def offset_list2str(list):
    return ';'.join(', '.join(map(str, offset)) for offset in list)

def offset_decreaseSentOffset(sentOffset, scu_offsets):
    return [[start_end[0] - sentOffset, start_end[1] - sentOffset] for start_end in scu_offsets]

# def Union(offsets, sentOffsets):
#     ranges_tmp = set([])
#     for offset, sentOffset in zip(offsets, sentOffsets):
#         offset = offset_str2list(offset)
#         offset = offset_decreaseSentOffset(sentOffset, offset)
#         ranges = [range(marking[0], marking[1]) for marking in offset]
#         ranges = set(chain(*ranges))
#         ranges_tmp = ranges_tmp | ranges
#     return  ranges_tmp


def Union(offsets):
    ranges_tmp = set([])
    for offset in offsets:
        offset = offset_str2list(offset)
        # offset = offset_decreaseSentOffset(sentOffset, offset)
        ranges = [range(marking[0], marking[1]) for marking in offset]
        ranges = set(chain(*ranges))
        ranges_tmp = ranges_tmp | ranges
    return  ranges_tmp

def char_idx2string(index_list, sentence):
    #input: index_list- list of indexes need to be selected from a sentence
    #       sentence- text string
    #output: the text span appears in the relevant indexes.
    index_list = sorted(list(index_list))
    if not index_list:  #if empty list
        return ''
    prev_char_idx = index_list[0]
    out_span = sentence[prev_char_idx]
    for char_idx in index_list[1:]:
        if char_idx != prev_char_idx +1:
            out_span += ' ' # add space before new span
        out_span += sentence[char_idx] if len(sentence) > char_idx else ''
        prev_char_idx = char_idx

    return out_span

def adjust_gold_indices(doc_path, gold_union_indices, doc_text):
    with open(doc_path, 'r') as f:
        gold_doc_text = f.read()

    offset = 0
    gold_union_indices_updated = []
    for idx in sorted(list(gold_union_indices)):
        if gold_doc_text[idx] == '\n' or gold_doc_text[idx] == ' ':
            gold_union_indices_updated.append(idx - offset)
            continue
        if gold_doc_text[idx] != doc_text[idx - offset]:
            doc_text_substring = doc_text[max(idx - offset-30,0): idx - offset+30]
            add_offset = doc_text_substring.find(gold_doc_text[idx:idx + 6])
            if add_offset == -1:
                add_offset = doc_text_substring.find(gold_doc_text[idx - 6:idx]) + 6
            if idx - offset-30 < 0:
                offset =  idx - add_offset
            else:
                offset = offset + 30 -add_offset
            # for temp_offset in range(-10,10):
            #     if gold_doc_text[idx] == doc_text[idx - offset - temp_offset]:
            #         offset += temp_offset
            #         break
            assert(gold_doc_text[idx] == doc_text[idx - offset])
        gold_union_indices_updated.append(idx - offset)

    return set(gold_union_indices_updated)







def aggregate_query_per_cluster(highlights_gold):
    cluster2query = {}
    for topic in highlights_gold['topic'].drop_duplicates().to_list():
        summary = read_generic_file(args.summaries_path+topic+'.txt')
        highlights_gold_topic = highlights_gold[highlights_gold['topic'] == topic]
        for cluster_idx in highlights_gold_topic['cluster_idx'].drop_duplicates().to_list():
            highlights_gold_topic_cluster = highlights_gold_topic[highlights_gold_topic['cluster_idx'] == cluster_idx]
            ranges_gold = Union(highlights_gold_topic_cluster['summarySpanOffsets'].drop_duplicates().to_list())
            query = ''
            prev_idx = -1
            for idx in sorted(ranges_gold):
                if idx != prev_idx+1:
                    query += ' '
                query += summary[idx]
                prev_idx = idx
            cluster2query[topic+'_'+str(cluster_idx)] = query

    highlights_gold['query'] = highlights_gold.apply(lambda x: cluster2query[x.topic+'_'+str(x.cluster_idx)], axis=1)
    return highlights_gold

##################################
######     main     ##############
##################################
if __name__ == "__main__":

    alignments_gold = read_gold()
    alignments_gold = alignments2clusters(alignments_gold)
    alignments_gold_w_query = aggregate_query_per_cluster(alignments_gold)
    alignments_gold_w_query.to_csv('data_alignments_gold_w_query_fixed.csv', index=False)





