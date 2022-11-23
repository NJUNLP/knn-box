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

from knnbox.common_utils import (
    global_vars,
    select_keys_with_pad_mask, 
    archs, 
    disable_model_grad, 
    enable_module_grad,
)
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import KernelSmoothedCombiner


@register_model("kernel_smoothed_knn_mt")
class KernelSmoothedKNNMT(TransformerModel):
    r"""
    The kernel smoothed knn-mt model.
    """    
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # when train metak network, we should disable other module's gradient
        # and only enable the combiner(metak network)'s gradient 
        if args.knn_mode == "train_kster":
            disable_model_grad(self)
            enable_module_grad(self, "combiner")
    
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "train_kster", "inference"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of adaptive knn-mt")
        parser.add_argument("--knn-combiner-path", type=str, metavar="STR", default="/home/",
                            help="The directory to save/load KernelSmoothedCombiner")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)") 
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with AdaptiveKNNMTDecoder
        """
        return KernelSmoothedKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


class KernelSmoothedKNNMTDecoder(TransformerDecoder):
    r"""
    The adaptive knn-mt Decoder, equipped with Datastore, Retriever and AdaptiveCombiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        else:
            # kernel smoothed knn-mt need keys as input of kster network
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["keys", "vals"])
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            if args.knn_mode == "train_kster":
                self.combiner = KernelSmoothedCombiner(probability_dim=len(dictionary), 
                                                        query_dim=args.decoder_output_dim)
            elif args.knn_mode == "inference":
                self.combiner = KernelSmoothedCombiner.load(args.knn_combiner_path)

                ## maybe you are confused because you can't find out the code about
                ## saving AdaptiveCombiner to args.knn_combiner_path directory.
                ## To save the combiner whenever we got a best checkpoint, I have to write some ugly code.
                ## you can find the code inside `fairseq.checkpoint_utils.save_checkpoint` function.
                ## In fact, we shouldn't modify the fairseq code, but there is no better way
                ## to save a single module instead of entire translation model.
            
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
            self.datastore["keys"].add(keys.half())
        
        elif self.args.knn_mode == "inference" or self.args.knn_mode == "train_kster":
            self.retriever.retrieve(x, return_list=["vals", "keys", "query", "distances"]) 

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
            calculate the knn probability based on retrieved resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == "inference" or self.args.knn_mode == "train_kster":
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)




r""" Define some kernel smoothed knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("kernel_smoothed_knn_mt", "kernel_smoothed_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("kernel_smoothed_knn_mt", "kernel_smoothed_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("kernel_smoothed_knn_mt", "kernel_smoothed_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("kernel_smoothed_knn_mt", "kernel_smoothed_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("kernel_smoothed_knn_mt", "kernel_smoothed_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("kernel_smoothed_knn_mt", "kernel_smoothed_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("kernel_smoothed_knn_mt", "kernel_smoothed_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("kernel_smoothed_knn_mt", "kernel_smoothed_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)
    

        

