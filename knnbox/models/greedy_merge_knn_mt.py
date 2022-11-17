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

from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner

# here must use relative path `.vanilla_knn_mt` instead of `knnbox.models.vanilla_knn_mt`
# otherwise fairseq will regist vanilla_knn_mt twice and cause an error.
from .vanilla_knn_mt import VanillaKNNMT, VanillaKNNMTDecoder


@register_model("greedy_merge_knn_mt")
class GreedyMergeKNNMT(VanillaKNNMT):
    r"""
    The GreedyMerge knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add greedy merge knn-mt related args here
        """
        VanillaKNNMT.add_args(parser)
        parser.add_argument("--pca-dim", type=int, metavar="N", default=256,
                            help="The expected target dimension of PCA")
        parser.add_argumet("--merge-neighbors-n", type=int, metavar="N", default=2,
                            help="merge how many neighbors when trim the datastore")

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with GreedyMergeKNNMTDecoder
        """
        return GreedyMergeKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

class GreedyMergeKNNMTDecoder(VanillaKNNMTDecoder):
    r"""
    The greedy merge knn-mt Decoder, equipped with knn datastore, retriever and combiner.

    GreedyMergeKNNMTDecoder inherited from VanillaKNNMTDecoder so that
    we needn't write forward(..) and get_normalized_probs(..) twice. 
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        # TransformerDecoder is GreedyMergeKNNMTDecoder's super super class
        # here we call TransformerDecoder's __init__ function
        TransformerDecoder.__init__(self, args, dictionary, embed_tokens, no_encoder_attn)

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = GreedyMergeDatastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        elif args.knn_mode == "inference":
            # when inference, we don't load the keys, use its faiss index is enough
            self.datastore = GreedyMergeDatastore.load(args.knn_datastore_path, exclude_load_list=["keys"])
            self.datastore.load_faiss_index("keys.faiss_index")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda,
                     temperature=args.knn_temperature, probability_dim=len(dictionary))




r""" Define some adaptive knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("greedy_merge_knn_mt", "greedy_merge_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("greedy_merge_knn_mt", "greedy_merge_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("greedy_merge_knn_mt", "greedy_merge_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("greedy_merge_knn_mt", "greedy_merge_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("greedy_merge_knn_mt", "greedy_merge_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("greedy_merge_knn_mt", "greedy_merge_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("greedy_merge_knn_mt", "greedy_merge_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("greedy_merge_knn_mt", "greedy_merge_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)
    

        

