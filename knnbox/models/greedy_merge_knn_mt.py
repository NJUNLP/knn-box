from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch
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
from knnbox.datastore import GreedyMergeDatastore
from knnbox.retriever import Retriever, CacheRetriever
from knnbox.combiner import Combiner, CacheCombiner, MergeWeightCombiner

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
        parser.add_argument("--do-pca", action="store_true", default=False,
                            help="whether to do pca operatiion for datastore")
        parser.add_argument("--pca-dim", type=int, metavar="N", default=256,
                            help="The expected target dimension of PCA")
        parser.add_argument("--do-merge", action="store_true", default=False,
                            help="whether to use greedy merge to prune the datastore")
        parser.add_argument("--merge-neighbors-n", type=int, metavar="N", default=2,
                            help="merge how many neighbors when trim the datastore")
        parser.add_argument("--enable-cache", action="store_true", default=False,
                            help="whether to use a retriever cache when inference.")
        parser.add_argument("--cache-threshold", type=float, default=6.0,
                            help="the threshold distance to use cache")
        parser.add_argument("--use-merge-weights", action="store_true", default=False,
                            help="whether to use merge weights when calclulate knn probs") 
        
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
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return GreedyMergeKNNMTEncoder(
            args,
            src_dict,
            embed_tokens
        )
    
class GreedyMergeKNNMTEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        # add next line
        self.args = args

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        ret = super().forward(src_tokens, src_lengths, return_all_hiddens, token_embeddings)

        # when encoder call forward, indicate a new batch comes
        if self.args.enable_cache:
            global_vars()["new_batch_comes"] = True

        return ret



class GreedyMergeKNNMTDecoder(TransformerDecoder):
    r"""
    The greedy merge knn-mt Decoder, equipped with knn datastore, retriever and combiner.

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
            load_list = ["vals"]
            if self.args.use_merge_weights:
                load_list.append("merge_weights")

            self.datastore = GreedyMergeDatastore.load(args.knn_datastore_path, 
                                load_list=load_list) 
            self.datastore.load_faiss_index("keys", move_to_gpu=False)

            if args.enable_cache:
                self.retriever = CacheRetriever(datastore=self.datastore, k=args.knn_k)
                self.combiner = CacheCombiner(lambda_=args.knn_lambda, 
                                temperature=args.knn_temperature, probability_dim=len(dictionary))
            else:
                self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
                if args.use_merge_weights:
                    self.combiner = MergeWeightCombiner(lambda_=args.knn_lambda,
                                temperature=args.knn_temperature, probability_dim=len(dictionary))
                else:
                    self.combiner = Combiner(lambda_=args.knn_lambda, 
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
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if self.args.knn_mode == "build_datastore":
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            # save half precision keys
            self.datastore["keys"].add(keys.half())
        
        elif self.args.knn_mode == "inference":
            if self.args.enable_cache:
                ## if a new batch comes, clear the CacheRetriever's cache
                if global_vars()["new_batch_comes"]:
                    self.retriever.clear_cache()
                global_vars()["new_batch_comes"] = False

            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            return_list = ["vals", "distances"]
            if self.args.enable_cache:
                return_list.append("query")
            if self.args.use_merge_weights:
                return_list.append("merge_weights")

            if self.args.enable_cache:
                self.retriever.retrieve(x, return_list=return_list, cache_threshold=self.args.cache_threshold)
            else:
                self.retriever.retrieve(x, return_list=return_list)

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
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, \
                        use_merge_weights=self.args.use_merge_weights, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)
        


r""" Define some greedy merge knn-mt's arch.
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
    

        

