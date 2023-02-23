from typing import Any, Dict, List, Optional, Tuple
import math
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

from knnbox.common_utils import (
    global_vars,
    select_keys_with_pad_mask,
    disable_model_grad,
    enable_module_grad,
    archs,
)
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import RobustCombiner


@register_model("robust_knn_mt")
class RobustKNNMT(TransformerModel):
    r"""
    The robust knn-mt model.
    """
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # when train metak network, we should disable other module's gradient
        # and only enable the combiner(metak network)'s gradient 
        if args.knn_mode == "train_metak":
            disable_model_grad(self)
            enable_module_grad(self, "combiner")

    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "train_metak", "inference"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-max-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter max k of robust knn-mt")
        
        # ! Robust kNN-MT has k, lambda and temperature trainable
        # parser.add_argument("--knn-k-type", choices=["fixed", "trainable"], default="trainable",
        #                     help="trainable k or fixed k, if choose `fixed`, we use all the"
        #                     "entries returned by retriever to calculate knn probs, "
        #                     "i.e. directly use --knn-max-k as k")
        # parser.add_argument("--knn-lambda-type", choices=["fixed", "trainable"], default="trainable",
        #                     help="trainable lambda or fixed lambda")
        # parser.add_argument("--knn-lambda", type=float, default=0.7,
        #                     help="if use a fixed lambda, provide it with --knn-lambda")
        # parser.add_argument("--knn-temperature-type", choices=["fixed", "trainable"], default="trainable",
        #                     help="trainable temperature or fixed temperature")
        # parser.add_argument("--knn-temperature", type=float, default=10,
        #                     help="if use a fixed temperature, provide it with --knn-temperature")
        parser.add_argument("--knn-combiner-path", type=str, metavar="STR", default="/home/",
                            help="The directory to save/load robustCombiner")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)") 
        
        # ? hyper-params for robust training 
        parser.add_argument("--robust-training-sigma", type=float, default=0.01,
                            help="the noise vector is sampled from a Gaussian distribution with variance sigma^2")
        parser.add_argument("--robust-training-alpha0", type=float, default=1.0,
                            help="alpha0 control the initial value of the perturbation ratio (alpha)")
        parser.add_argument("--robust-training-beta", type=int, default=1000,
                            help="beta control the declining speed of the perturbation ratio (alpha)")
        # ? hyper-params for DC & WP networks
        parser.add_argument("--robust-dc-hidden-size", type=int, default=4,
                            help="the hidden size of DC network")
        parser.add_argument("--robust-wp-hidden-size", type=int, default=32,
                            help="the hidden size of WP network")
        parser.add_argument("--robust-wp-topk", type=int, default=8,
                            help="WP network uses the k highest probabilities of the NMT distribution as input")
        
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with RobustKNNMTDecoder
        """
        return RobustKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            target: Optional[Tensor] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            target=target,
        )
        return decoder_out


class RobustKNNMTDecoder(TransformerDecoder):
    r"""
    The robust knn-mt Decoder, equipped with Datastore, Retriever and RobustCombiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.update_num = 0

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        else:
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["keys", "vals"])
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            if args.knn_mode == "train_metak":
                self.combiner = RobustCombiner(
                    max_k=args.knn_max_k, 
                    midsize=args.robust_wp_hidden_size, 
                    midsize_dc=args.robust_dc_hidden_size, 
                    topk_wp=args.robust_wp_topk, 
                    probability_dim=len(dictionary)
                )
            elif args.knn_mode == "inference":
                self.combiner = RobustCombiner.load(args.knn_combiner_path)

                ## maybe you are confused because you can't find out the code about
                ## saving RobustCombiner to args.knn_combiner_path directory.
                ## To save the combiner whenever we got a best checkpoint, I have to write some ugly code.
                ## you can find the code inside `fairseq.checkpoint_utils.save_checkpoint` function.
                ## In fact, knnbox shouldn't modify the fairseq code, but there is no better way
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
        target: Optional[Tensor] = None,
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
        
        elif self.args.knn_mode == "inference" or self.args.knn_mode == "train_metak":
            self.retriever.retrieve(x, return_list=["vals", "query", "distances", "keys"]) 

        extra.update({"last_hidden": x, "target": target})
        
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
        if self.args.knn_mode == "inference" or self.args.knn_mode == "train_metak":
            network_probs = utils.softmax(net_output[0], dim=-1, onnx_trace=self.onnx_trace)
            knn_dists = self.retriever.results["distances"]
            tgt_index = self.retriever.results["vals"]
            knn_key = self.retriever.results["keys"]
            queries = self.retriever.results["query"]
            
            knn_dists = torch.sum((knn_key - queries.unsqueeze(-2).detach()) ** 2, dim=-1)   
            knn_dists, new_index = torch.sort(knn_dists, dim=-1)
            tgt_index = tgt_index.gather(dim=-1, index=new_index)
            knn_key = knn_key.gather(dim=-2, index=new_index.unsqueeze(-1).expand(knn_key.shape))
            
            B, S, K = knn_dists.size()
            network_select_probs = network_probs.gather(index=tgt_index, dim=-1) # [batch, seq len, K]
            
            if self.training:
                target=net_output[1]["target"]
                last_hidden=net_output[1]["last_hidden"]
                random_rate = self.args.robust_training_alpha0
                noise_var = self.args.robust_training_sigma
                e = self.args.robust_training_beta
                random_rate = random_rate * math.exp((-self.update_num)/e)

                noise_mask = (tgt_index == target.unsqueeze(-1)).any(-1, True)
                rand_mask = ((torch.rand(B, S, 1).cuda() < random_rate) & (target.unsqueeze(-1) != 1)).long()
                rand_mask2 = ((torch.rand(B, S, 1).cuda() < random_rate) & (target.unsqueeze(-1) != 1) & ~noise_mask).float()
                                
                with torch.no_grad():
                    # add perturbation
                    knn_key = knn_key + torch.randn_like(knn_key) * rand_mask.unsqueeze(-1) * noise_var
                    new_key = last_hidden + torch.randn_like(last_hidden) * noise_var
                    noise_knn_key = torch.cat([new_key.unsqueeze(-2), knn_key.float()[:, :, :-1, :]], -2)
                    noise_tgt_index = torch.cat([target.unsqueeze(-1), tgt_index[:, :, :-1]], -1)               
                    tgt_index = noise_tgt_index * rand_mask2.long() + tgt_index * (1 - rand_mask2.long())
                    knn_key = noise_knn_key * rand_mask2.unsqueeze(-1) + knn_key * (1 - rand_mask2.unsqueeze(-1))
                    
                    knn_probs = utils.softmax(self.output_layer(knn_key.float()), dim=-1, onnx_trace=self.onnx_trace) # B, S, K, V
                    knn_key_feature = knn_probs.gather(-1, index=tgt_index.unsqueeze(-1)).squeeze(-1)
                    noise_knn_dists = torch.sum((knn_key - last_hidden.unsqueeze(-2).detach()) ** 2, dim=3)
                    dup_knn_dists = noise_knn_dists 

                    # sort the distance again
                    new_dists, dist_index = torch.sort(dup_knn_dists, dim=-1)
                    new_index = dist_index

                    # update the input
                    knn_dists = new_dists
                    tgt_index = tgt_index.gather(-1, new_index)
                    network_select_probs = network_probs.gather(index=tgt_index, dim=-1)
                    knn_key_feature = knn_key_feature.gather(-1, new_index)
                    
            else:
                knn_probs = utils.softmax(self.output_layer(knn_key.float()), dim=-1, onnx_trace=self.onnx_trace) # B, S, K, V
                knn_key_feature = knn_probs.gather(-1, index=tgt_index.unsqueeze(-1)).squeeze(-1)

            knn_prob = self.combiner.get_knn_prob(
                tgt_index=tgt_index,
                knn_dists=knn_dists,
                knn_key_feature=knn_key_feature,
                network_probs=network_probs,
                network_select_probs=network_select_probs,
                device=net_output[0].device
            )
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self.update_num = num_updates



r""" Define some robust knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("robust_knn_mt", "robust_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("robust_knn_mt", "robust_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("robust_knn_mt", "robust_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("robust_knn_mt", "robust_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("robust_knn_mt", "robust_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("robust_knn_mt", "robust_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("robust_knn_mt", "robust_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("robust_knn_mt", "robust_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)
        

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion("label_smoothed_cross_entropy_for_robust")
class LabelSmoothedCrossEntropyCriterionForRobust(
    LabelSmoothedCrossEntropyCriterion
):
    # ? label_smoothed_cross_entropy_for_robust is a CE-loss that passes target to model, which is required by robust training
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"], target=sample['target'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
