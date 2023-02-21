import os

import elasticsearch
import elasticsearch.helpers as helpers
import argparse
from tqdm import tqdm

from fairseq import utils
from fairseq.data import data_utils
from fairseq.tasks.translation import load_langpair_dataset, TranslationTask

def delete_index(es, index_name):
    r""" delete an index if exist"""
    try:
        result = es.indices.delete(index=index_name)
        print("> delete old {}".format(index_name))
    except elasticsearch.NotFoundError:
        print("> elasticsearch has no index name `{}`, we will create one".format(index_name))
  

def create_index(es, index_name):
    print("> start create index {}".format(index_name))
    mappings = {
        "settings": {
            "similarity": {
                "my_bm25": {
                    "type": "BM25",
                    "b": 0.75,  # default 
                    "k1": 1.2,  # default
                }
            }
        },
        "mappings": {
            "properties": {
                "data_id": {
                    "type": "integer", # the sentence index in datset
                    "index": "false", 
                },
                # source_tokens and target_tokens are sequence of 
                # token ids, such as "12 55 66 8", so use 
                # whitespace as tokenizer.
                "source_tokens": {
                    "type": "text",
                    "index": "true",
                    "analyzer": "whitespace",
                    "search_analyzer": "whitespace",
                },
                "target_tokens": {
                    "type": "text",
                    "index": "false",
                    "analyzer": "whitespace",
                    "search_analyzer": "whitespace",
                }
            }
            
        }
    }
    result = es.indices.create(index=index_name, body=mappings)
    print(result)
    print("Finish create <<<")
    
def insert_content(es, index_name, dataset):
    r""" insert content to es index """
    print("start insert content to {}".format(index_name))
    data_list = []
    insert_every = 10000   # collect every 1w sentences, insert to es
    
    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description("Inserting: ")
        for data_id, data in enumerate(dataset):
            source_tokens = data["source"][:-1] \
                if data["source"][-1] == dataset.eos else data["source"] # remove <eos>
            target_tokens = data["target"][:-1] \
                if data["target"][-1] == dataset.eos else data["target"] # remove <eos>
            # convert to string
            source_tokens = " ".join([str(num) for num in source_tokens.numpy().tolist()])
            target_tokens = " ".join([str(num) for num in target_tokens.numpy().tolist()])
            data = {
                "data_id": data_id,
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
            }
            data_list.append(data)
            if (data_id + 1) % insert_every == 0 or data_id == len(dataset) - 1:
                # batched insert
                actions = [
                    {
                        "_op_type": "index",
                        "_index": index_name,
                        "_source": d,
                    }
                    for d in data_list
                ]
                data_list.clear()
                helpers.bulk(es, actions)
                # es.index(index=index_name, body=data)
                insert_size = insert_every if data_id != len(dataset) - 1 else \
                                len(dataset) % insert_every
                pbar.update(insert_size)
    print("Finish inserting <<<")

def load_dataset(args):
    r""" load the fairseq dataset """
    # find language pair automatically
    paths = utils.split_paths(args.dataset_path)
    source_lang, target_lang = data_utils.infer_language_pair(paths[0])
    # load dictionaries
    src_dict = TranslationTask.load_dictionary(
                os.path.join(paths[0], "dict.{}.txt").format(source_lang))
    tgt_dict = TranslationTask.load_dictionary(
                os.path.join(paths[0], "dict.{}.txt".format(target_lang))
    )
    assert src_dict.pad() == tgt_dict.pad()
    assert src_dict.eos() == tgt_dict.eos()
    assert src_dict.unk() == tgt_dict.unk()
    # load dataset
    dataset = load_langpair_dataset(
        data_path = args.dataset_path,
        split = "train",    # load the train set
        src = source_lang,
        src_dict = src_dict,
        tgt = target_lang,
        tgt_dict = tgt_dict,
        combine = False,
        dataset_impl = args.dataset_impl,
        upsample_primary = 1.0,
        left_pad_source = False, 
        left_pad_target = False, 
        max_source_positions = 1024,
        max_target_positions = 1024,
        load_alignments = False,
        truncate_source = False,
    )
    return dataset

if __name__ == "__main__":
    # parse command args
    parser = argparse.ArgumentParser("build_elasticsearch_database")
    parser.add_argument("--dataset-path", type=str, help="The dataset path")
    parser.add_argument("--dataset-impl", type=str, default="mmap", help="the format of the dataset")
    parser.add_argument("--elastic-index-name", type=str, help="The index name which use")
    parser.add_argument("--elastic-port", type=int, default=9200, help="the port of elastic service")
    args = parser.parse_args()

    # connect to ES server
    es = elasticsearch.Elasticsearch(['http://127.0.0.1:{}'.format(args.elastic_port)], request_timeout=3600)
    
    # laod dataset
    dataset = load_dataset(args)
    # delete index if already exist
    delete_index(es, args.elastic_index_name)
    # create index
    create_index(es, args.elastic_index_name)
    # # insert content
    insert_content(es, args.elastic_index_name, dataset)

    

