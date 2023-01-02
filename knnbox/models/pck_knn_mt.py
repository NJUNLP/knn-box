from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch
import torch.nn.functional as F
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
from knnbox.datastore import Datastore, PckDatastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner, AdaptiveCombiner
from .adaptive_knn_mt import AdaptiveKNNMT


@register_model("pck_knn_mt")
class PckKNNMT(AdaptiveKNNMT):
    r"""
    The  pck knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add pck knn-mt related args here
        """
        AdaptiveKNNMT.add_args(parser)
        parser.add_argument("--knn-reduct-dim", type=int, metavar="N", default=64,
                            help="reducted dimension of datastore")
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with PckKNNMTDecoder
        """
        return PckKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


class PckKNNMTDecoder(TransformerDecoder):
    r"""
    The pck knn-mt Decoder, equipped with knn datastore, retriever and combiner.
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
                global_vars()["datastore"] = PckDatastore(
                        path=args.knn_datastore_path,
                        dictionary_len=len(self.dictionary),
                    )  
            self.datastore = global_vars()["datastore"]
        
        else:
            self.datastore = PckDatastore.load(args.knn_datastore_path, load_list=["vals"], load_network=True)
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            if args.knn_mode == "train_metak":
                self.combiner = AdaptiveCombiner(max_k=args.knn_max_k, probability_dim=len(dictionary),
                            k_trainable=(args.knn_k_type=="trainable"),
                            lambda_trainable=(args.knn_lambda_type=="trainable"), lamda_=args.knn_lambda,
                            temperature_trainable=(args.knn_temperature_type=="trainable"), temperature=args.knn_temperature
            )
            elif args.knn_mode == "inference":
                self.combiner = AdaptiveCombiner.load(args.knn_combiner_path)

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
            def get_4_gram(target):
                r"""
                Args:
                    target: [B, T]
                Return: [B, T, 4]
                """
                batch_size = target.size(0)
                target = target[:, :, None]
                target_pad_1 = torch.cat((torch.zeros((batch_size, 1, 1), device=x.device, dtype=torch.long), target[:, :-1]), 1)
                target_pad_2 = torch.cat((torch.zeros((batch_size, 2, 1), device=x.device, dtype=torch.long), target[:,:-2]), 1)
                target_pad_3 = torch.cat((torch.zeros((batch_size, 3, 1), device=x.device, dtype=torch.long), target[:,:-3]), 1)
                return torch.cat((target, target_pad_1, target_pad_2, target_pad_3), -1)
            
            def get_tgt_probs(probs, target):
                r""" 
                Args:
                    probs: [B, T, dictionary]
                    target: [B, T]
                Return: [B, T]
                """
                B, T, C = probs.size(0), probs.size(1), probs.size(2)
                one_hot = torch.arange(0, C).to(target.device)[None, None].repeat(B, T, 1) == target[:, :, None]
                return (probs * one_hot.float()).sum(-1)

            def get_4_gram_probs(target_prob):
                r"""
                Args:
                    target_prob: [B, T]
                Return: [B, T, 4]
                """
                target_prob = target_prob[:, :, None]
                target_pad_1 = torch.cat((target_prob[:, :1].repeat(1, 1, 1), target_prob[:, :-1]), 1)
                target_pad_2 = torch.cat((target_prob[:, :1].repeat(1, 2, 1), target_prob[:, :-2]), 1)
                target_pad_3 = torch.cat((target_prob[:, :1].repeat(1, 3, 1), target_prob[:, :-3]), 1)
                return torch.cat((target_prob, target_pad_1,  target_pad_2, target_pad_3), -1)

            def get_entropy(probs):
                r"""probs: [B, T, dictionary]"""
                return - (probs * torch.log(probs+1e-7)).sum(-1)

            # calulate probs
            output_logit = self.output_layer(x) 
            output_probs = F.softmax(output_logit, dim=-1)
            # get useful info
            target = self.datastore.get_target()
            ids_4_gram = get_4_gram(target) # [B, T, 4]
            target_prob = get_tgt_probs(output_probs, target) # [B, T]
            probs_4_gram = get_4_gram_probs(target_prob) # [B, T, 4]
            entropy = get_entropy(output_probs) # [B, T]
            # process pad 
            pad_mask = self.datastore.get_pad_mask()
            keys = select_keys_with_pad_mask(x, pad_mask)
            ids_4_gram = select_keys_with_pad_mask(ids_4_gram, pad_mask)
            probs_4_gram = select_keys_with_pad_mask(probs_4_gram, pad_mask)
            entropy = entropy.masked_select(pad_mask) 
            # save infomation to datastore
            self.datastore["keys"].add(keys.half())
            self.datastore["ids_4_gram"].add(ids_4_gram)
            self.datastore["probs_4_gram"].add(probs_4_gram)
            self.datastore["entropy"].add(entropy) 

        elif self.args.knn_mode == "train_metak" or self.args.knn_mode == "inference":
            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            self.retriever.retrieve(self.datastore.vector_reduct(x), return_list=["vals", "distances"])
        
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
        if self.args.knn_mode == "inference" or self.args.knn_mode == "train_metak":
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


r""" Define some pck knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("pck_knn_mt", "pck_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

    
    

        

