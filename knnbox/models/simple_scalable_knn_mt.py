import torch
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.data import data_utils
from fairseq import utils

import elasticsearch
import editdistance
import numpy as np

from knnbox.common_utils import global_vars, filter_pad_tokens, select_keys_with_pad_mask, archs
from knnbox.retriever import SimpleScalableRetriever
from knnbox.combiner import SimpleScalableCombiner
# from knnbox.combiner import Combiner


@register_model("simple_scalable_knn_mt")
class SimpleScalableKNNMT(TransformerModel):
    r"""
    The simple and scalable knn-mt model.
    """
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        if "sk_mt_model" not in global_vars():
            global_vars()["sk_mt_model"] = self


    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["inference"],
                            help="choose the action mode")
        parser.add_argument("--knn-k", type=int, metavar="N", default=2,
                            help="The hyper-parameter k of  knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=100,
                            help="The hyper-parameter temperature of  knn-mt")
        parser.add_argument("--reserve-top-m", type=int, default=16,
                            help="reserve top-m retrieve results")
        parser.add_argument("--elastic-index-name", type=str, help="The elasticsearch \
                            index name which to retrieve from")
        parser.add_argument("--elastic-port", type=int, default=9200, 
                            help="The port of elasticsearch service.")
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with SimpleScalableKNNMTEncoder
        """
        return SimpleScalableKNNMTEncoder(args, src_dict, embed_tokens)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with SimpleScalableKNNMTDecoder
        """
        return SimpleScalableKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
    
def vector_hash_func(x):
    r""" Hash torch.tensor
    Args:
        x: [batch, seq_len, dim]
    Return:
        str[batch]
    """
    return np.around(x[:,0,0].cpu().numpy(), decimals=6)


    
def collate(source, target, pad_idx=1, eos_idx=2, left_pad_source=True, left_pad_target=False):
    r"""collate tokens to get a mini-batch
    Args:
        source: list of torch.tensor
        target: list of torch.tensor
    """
    def merge(tokens, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            tokens,
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=None,
            pad_to_multiple=1,
        )
    src_tokens = merge(tokens=source, left_pad=left_pad_source, 
                        move_eos_to_beginning=False)
    # sort by descending source length
    src_lengths = torch.LongTensor(
       [s.ne(pad_idx).long().sum() for s in source]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    src_tokens = src_tokens.index_select(0, sort_order)
    # TODO: use left_pad_target instead of False
    tgt_tokens = merge(tokens=target, left_pad=False, move_eos_to_beginning=False)
    tgt_tokens = tgt_tokens.index_select(0, sort_order)

    prev_output_tokens = merge(tokens=target, left_pad=False,
                        move_eos_to_beginning=True)
    prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

    return {"src_tokens": src_tokens, "src_lengths": src_lengths,
            "tgt_tokens": tgt_tokens, "prev_output_tokens": prev_output_tokens,
            "sort_order": sort_order}



class SimpleScalableKNNMTEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        encoder_out = super().forward(src_tokens, src_lengths, return_all_hiddens, token_embeddings)
        if self.args.knn_mode == "inference":
            with torch.no_grad():
                self._build_dynamic_datastore(encoder_out, src_tokens)
        return encoder_out
    
    def _build_dynamic_datastore(self, encoder_out, src_tokens):
        r""" build dynamic datastore when forwarding encoder
        Args: 
            encoder_out
            src_tokens
        Return: mapping from encoder_out_hash to (key vector, val vector)
        """
        
        es = elasticsearch.Elasticsearch(
            ["http://localhost:"+str(self.args.elastic_port)], request_timeout=3600)
        # disable es logging to stdout
        tracer = logging.getLogger("elasticsearch")
        tracer.setLevel(logging.CRITICAL)   
        # step 1. use source tokens to query elasticsearch database
        requests = []
        for src in src_tokens:
            src = [str(token_id) for token_id in src.cpu().numpy().tolist() if token_id != 1 and token_id != 2]
            src = " ".join(src)
            # req head
            req_head = {"index": self.args.elastic_index_name}
            # req body
            req_body = {"query": {"match": {"source_tokens": src}}, "from":0, "size": 64}
            requests.extend([req_head, req_body])

        resp = es.msearch(body=requests)["responses"]

        retrieve_results = [None]*len(src_tokens)
        for i in range(len(src_tokens)):
            hits = resp[i]["hits"]["hits"]
            retrieve_results[i] = [[], []]
            for sentence in hits:
                retrieve_results[i][0].append(sentence["_source"]["source_tokens"])
                retrieve_results[i][1].append(sentence["_source"]["target_tokens"]) 
        # step 2. re-rank with similarity
        # similarity(x, xi) = 1 - dist(x, xi)/max(|x|, |xi|)
        for i in range(len(src_tokens)):
            similarities = []
            for retrieve_src in retrieve_results[i][0]:
                retrieve_src_list =  [int(num) for num in retrieve_src.split()]
                edit_distance = editdistance.eval([tok for tok in src_tokens[i].cpu().numpy().tolist()[:-1] if tok != 1 and tok != 2], # drop <eos>
                                    retrieve_src_list)
                similarities.append(1. - float(edit_distance)
                        / max(len(src_tokens[i])-1, len(retrieve_src_list)))

            # sort by similary, descending order
            indices = np.argsort(-np.array(similarities))
            retrieve_results[i][0] = [
                retrieve_results[i][0][idx] for idx in indices[:self.args.reserve_top_m]
            ]
            retrieve_results[i][1] = [
                retrieve_results[i][1][idx] for idx in indices[:self.args.reserve_top_m]
            ]


        # step 3. forward the model.
        # TODO: use bigger batch to speed up
        encoder_out_hash = vector_hash_func(encoder_out[0].transpose(0,1))
        global_vars()["encoderout_to_kv"] = {}
        for i in range(len(src_tokens)):
            if len(retrieve_results[i][0]) == 0:
                global_vars()["encoderout_to_kv"][float(encoder_out_hash[i])] = None
                print("---- WARNING: Here a sentence retrieved empty result: -----")
                print("     > src: ", self.dictionary.string(src_tokens[i]))
                continue
            # construct the model input
            source =  [torch.LongTensor([int(token) for token in s.split()]+[self.dictionary.eos()])
                        for s in retrieve_results[i][0]] 
    
            target = [torch.LongTensor([int(token) for token in s.split()]+[self.dictionary.eos()])
                        for s in retrieve_results[i][1]]
            
            batch = collate(source, target, pad_idx=self.dictionary.pad(),
                            eos_idx=self.dictionary.eos(),
                            left_pad_source=self.args.left_pad_source, 
                            left_pad_target=self.args.left_pad_target)
            if torch.cuda.is_available() and not self.args.cpu:
                batch = utils.move_to_cuda(batch)   # move to cuda
            # modify knn_mode to prevent infinite function call
            global_vars()["sk_mt_model"].args.knn_mode = "-" 
            model_decoder_out = global_vars()["sk_mt_model"](
                batch["src_tokens"],
                batch["src_lengths"],
                batch["prev_output_tokens"],
                return_all_hiddens=False,
                features_only=True,
            )[0]
         
            # recover knn_mode
            global_vars()["sk_mt_model"].args.knn_mode = "inference"
            # vals
            non_pad_tokens, mask = filter_pad_tokens(batch["tgt_tokens"])
            # keys
            keys = select_keys_with_pad_mask(model_decoder_out, mask)
            
            global_vars()["encoderout_to_kv"][float(encoder_out_hash[i])] = \
                {"keys": keys, "vals": non_pad_tokens}



class SimpleScalableKNNMTDecoder(TransformerDecoder):
    r"""
    The simple and scalable knn-mt Decoder """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        if args.knn_mode == "inference":
            # we access elasticsearch database through port 9200,
            # so here need no regular datastore
            self.retriever = SimpleScalableRetriever(k=args.knn_k, )
            self.combiner = SimpleScalableCombiner(
                temperature=args.knn_temperature, probability_dim=len(dictionary))

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `inference`, we retrieve the elasticsearch database
        with source tokens.
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if self.args.knn_mode == "inference":
            # 
            encoder_out_hash = vector_hash_func(encoder_out[0].transpose(0,1))
            # we will use the encoder_out_hash to get dynamic_datastore
            self.retriever.retrieve(query=x, encoder_out_hash=encoder_out_hash,
                        return_list=["vals", "distances"])
        
        if not features_only:
            x = self.output_layer(x)
        return x, extra
    

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == "inference":
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


r""" Define some simple scalable knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

@register_model_architecture("simple_scalable_knn_mt", "simple_scalable_knn_mt@transformer_zh_en")
def transformer_zh_en(args):
    archs.transformer_zh_en(args)
