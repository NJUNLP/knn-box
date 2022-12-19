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


@register_model("plac_knn_mt")
class PlacKNNMT(TransformerModel):
    r"""
    The plac knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["save_mt_pred"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with PlacKNNMTDecoder
        """
        return PlacKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


class PlacKNNMTDecoder(TransformerDecoder):
    r"""
    The plac knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        assert args.knn_mode == "save_mt_pred", 'PLAC Model can only be used to save mt model predictions'
        if "datastore" not in global_vars():
            # regist the datastore as a global variable if not exist,
            # because we need access the same datastore in another 
            # python file (when traverse the dataset and `add value`)
            global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
        self.datastore = global_vars()["datastore"]

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

        assert self.args.knn_mode == "save_mt_pred", 'PLAC Model can only be used to save mt model predictions'
        keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
        # save half precision keys
        self.datastore["keys"].add(keys.half())
        
        if not features_only:
            x = self.output_layer(x)
        mt_preds = x.argmax(dim=-1, keepdim=True)
        mt_preds = select_keys_with_pad_mask(mt_preds, self.datastore.get_pad_mask())
        self.datastore["mt_preds"].add(mt_preds.squeeze(-1))

        return x, extra


r""" Define some plac knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("plac_knn_mt", "plac_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("plac_knn_mt", "plac_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("plac_knn_mt", "plac_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("plac_knn_mt", "plac_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("plac_knn_mt", "plac_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("plac_knn_mt", "plac_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("plac_knn_mt", "plac_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("plac_knn_mt", "plac_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)
