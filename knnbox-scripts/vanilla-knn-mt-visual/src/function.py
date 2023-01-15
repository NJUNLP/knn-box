from PIL import Image
import numpy as np
import json
import os
import altair as alt
import pandas as pd
from fairseq.data.dictionary import Dictionary
import streamlit as st
import yaml
from sklearn.decomposition import PCA

import fileinput
import logging
import math
import os
import sys
import time
from collections import namedtuple
import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
from fairseq import metrics, search, tokenizer, utils
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators
from fairseq.dataclass.utils import gen_parser_from_dataclass
from knn_sequence_generator import KNNSequenceGenerator

from knnbox.common_utils import Memmap

@st.cache(allow_output_mutation=True)
def get_icon():
    icon = Image.open("./src/logo_transparent.png")
    return icon


@st.cache(allow_output_mutation=True)
def get_config():
    with open("./model_configs.yml") as f:
        cfg = yaml.safe_load(f)
    return cfg


@st.cache(allow_output_mutation=True)
def get_datastore_cfgs(datastore_path):
    with open(os.path.join(datastore_path, "config.json")) as file:
        config = json.load(file)
    return config


@st.cache(allow_output_mutation=True, max_entries=5)
def get_spatial_distribution(datastore_path, dictionary_path, sample_nums=30000, max_entries=10):
    r"""
    Return chart
    """
    with open(os.path.join(datastore_path, "config.json")) as file:
        config = json.load(file)
    # load keys and values
    keys = Memmap(os.path.join(datastore_path, "keys.npy"), 
            dtype=config["data_infos"]["keys"]["dtype"], shape=config["data_infos"]["keys"]["shape"])
    values = Memmap(os.path.join(datastore_path, "vals.npy"),
            dtype=config["data_infos"]["vals"]["dtype"], shape=config["data_infos"]["vals"]["shape"])
    sample_indices = np.random.choice(config["data_infos"]["keys"]["shape"][0], size=sample_nums, replace=False)

    sampled_keys = keys.data[sample_indices]
    sample_values = values.data[sample_indices]
    dictionary = Dictionary.load(dictionary_path)
    words = [dictionary[i] for i in sample_values]

    # PCA key to 2 dimesion
    pca = PCA(n_components=2)
    pca.fit(sampled_keys)
    pca_sampled_keys = pca.transform(sampled_keys)

    df = pd.DataFrame({
        "x": pca_sampled_keys[:,0],
        "y": pca_sampled_keys[:,1],
        "value": words,
    })
    
    selector = alt.selection_single(fields=["value"])
    chart = alt.Chart(df).mark_circle(size=100).encode(
        x = "x",
        y = "y",
        color = alt.condition(selector, "value", alt.value("transparent")),
        # color = "value",
        tooltip = ["x", "y", "value"],
    ).add_selection(selector).interactive()
    
    # release memory
    del keys
    del values
    del sampled_keys
    del sample_values
    del words
    del dictionary

    return chart


def display_partial_records(frequency_records, ratio, display_sz=20): 
    dictionary_sz = len(frequency_records)

    start_idx = int(dictionary_sz*ratio)
    end_idx = min(int(start_idx+display_sz), dictionary_sz)

    frequency = []
    word = []
    for i in range(start_idx, end_idx):
        word.append(frequency_records[i][0])
        frequency.append(frequency_records[i][1])

    ds = pd.DataFrame({
        "frequency": frequency,
        "word": word,
    })

    bars = alt.Chart(ds).mark_bar().encode(
        x=alt.X('frequency', sort=None),
        y=alt.Y("word",sort=None),
    )

    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=5  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='frequency'
    )

    return (bars + text).properties(height=500)



@st.cache(allow_output_mutation=True, max_entries=5)
def get_value_frequency(datastore_path, dictionary_path):
    with open(os.path.join(datastore_path, "config.json")) as file:
        config = json.load(file)

    # load values
    values = Memmap(os.path.join(datastore_path, "vals.npy"),
            dtype=config["data_infos"]["vals"]["dtype"], shape=config["data_infos"]["vals"]["shape"])
    dictionary = Dictionary.load(dictionary_path)

    records = {}
    for i in values.data:
        word = dictionary[i]
        if word not in records:
            records[word] = 1
        else:
            records[word] += 1

    record_list = []
    for k, v in records.items():
        record_list.append([k, v])
    record_list.sort(key=lambda r: r[1], reverse=True)
    # sort by frequency
    return record_list



def knn_build_generator(models, args, target_dictionary, seq_gen_cls=None, extra_gen_cls_kwargs=None):
    if getattr(args, "score_reference", False):
        from fairseq.sequence_scorer import SequenceScorer

        return SequenceScorer(
            target_dictionary,
            compute_alignment=getattr(args, "print_alignment", False),
        )

    from fairseq.sequence_generator import (
        SequenceGenerator,
        SequenceGeneratorWithAlignment,
    )

    # Choose search strategy. Defaults to Beam Search.
    sampling = getattr(args, "sampling", False)
    sampling_topk = getattr(args, "sampling_topk", -1)
    sampling_topp = getattr(args, "sampling_topp", -1.0)
    diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
    diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
    match_source_len = getattr(args, "match_source_len", False)
    diversity_rate = getattr(args, "diversity_rate", -1)
    constrained = getattr(args, "constraints", False)
    prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
    if (
        sum(
            int(cond)
            for cond in [
                sampling,
                diverse_beam_groups > 0,
                match_source_len,
                diversity_rate > 0,
            ]
        )
        > 1
    ):
        raise ValueError("Provided Search parameters are mutually exclusive.")
    assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
    assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

    if sampling:
        search_strategy = search.Sampling(
            target_dictionary, sampling_topk, sampling_topp
        )
    elif diverse_beam_groups > 0:
        search_strategy = search.DiverseBeamSearch(
            target_dictionary, diverse_beam_groups, diverse_beam_strength
        )
    elif match_source_len:
        # this is useful for tagging applications where the output
        # length should match the input length, so we hardcode the
        # length constraints for simplicity
        search_strategy = search.LengthConstrainedBeamSearch(
            target_dictionary,
            min_len_a=1,
            min_len_b=0,
            max_len_a=1,
            max_len_b=0,
        )
    elif diversity_rate > -1:
        search_strategy = search.DiverseSiblingsSearch(
            target_dictionary, diversity_rate
        )
    elif constrained:
        search_strategy = search.LexicallyConstrainedBeamSearch(
            target_dictionary, args.constraints
        )
    elif prefix_allowed_tokens_fn:
        search_strategy = search.PrefixConstrainedBeamSearch(
            target_dictionary, prefix_allowed_tokens_fn
        )
    else:
        search_strategy = search.BeamSearch(target_dictionary)

    if seq_gen_cls is None:
        if getattr(args, "print_alignment", False):
            seq_gen_cls = SequenceGeneratorWithAlignment
        else:
            seq_gen_cls = KNNSequenceGenerator
    extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
    return seq_gen_cls(
        models,
        target_dictionary,
        beam_size=getattr(args, "beam", 5),
        max_len_a=getattr(args, "max_len_a", 0),
        max_len_b=getattr(args, "max_len_b", 200),
        min_len=getattr(args, "min_len", 1),
        normalize_scores=(not getattr(args, "unnormalized", False)),
        len_penalty=getattr(args, "lenpen", 1),
        unk_penalty=getattr(args, "unkpen", 0),
        temperature=getattr(args, "temperature", 1.0),
        match_source_len=getattr(args, "match_source_len", False),
        no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
        search_strategy=search_strategy,
        **extra_gen_cls_kwargs,
    )


def get_knn_interactive_generation_parser(interactive=True, default_task="translation"):
    r""" 
    modify the options.get_interacitve_generation_parser()
    functions to parse args
    """
    parser = options.get_parser("Generation", default_task)
    options.add_dataset_args(parser, gen=True)
    options.add_distributed_training_args(parser, default_world_size=1)
    options.add_model_args(parser)
    options.add_generation_args(parser)
    if interactive:
        options.add_interactive_args(parser)
    return parser


@st.cache(allow_output_mutation=True)
def get_knn_model_resource(
    data_path,
    model_path,
    arch,
    knn_mode,
    knn_datastore_path,
    user_dir,
    bpe,
    bpe_codes,
    beam,
    lenpen,
    max_len_a,
    max_len_b,
    source_lang,
    target_lang,
    max_tokens,
    scoring,
    tokenizer,
):
    r""" use st.cache decorator when load model and other resources """

    # to use fairseq, we pseudo cmd args here
    pseudo_args = [
        "QAQ~", data_path,
        "--arch", arch,
        "--knn-mode", knn_mode,
        "--knn-k", "8",
        "--knn-lambda", "0.0",
        "--knn-temperature", "10.0",
        "--user-dir", user_dir,
        "--knn-datastore-path", knn_datastore_path,
        "--path", model_path,
        "--beam", beam,
        "--lenpen", lenpen,
        "--max-len-a", max_len_a,
        "--max-len-b", max_len_b,
        "--source-lang", source_lang,
        "--target-lang", target_lang,
        "--max-tokens", max_tokens,
        "--scoring", scoring,
        "--tokenizer", "space",  # we have already tokenize in "app.py", here we use space tokenizer is enough
        "--task", "translation",
        "--bpe", bpe,
        "--bpe-codes", bpe_codes,
        "--nbest", "1",
        "--remove-bpe",
    ]
    # argparse acutally read sys.argv, so we append our args to it
    # use unittest.mock.patch, we limit the scope of appended args
    from unittest.mock import patch
    with patch("sys.argv", pseudo_args):
        import sys
        print(sys.argv)
        parser = get_knn_interactive_generation_parser()
        args = options.parse_args_and_arch(parser)
        # parse twice
        override_parser = get_knn_interactive_generation_parser()
        override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)
    
    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    
    # import the module
    utils.import_user_module(args)
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.batch_size is None:
        args.batch_size = 1

    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not args.batch_size or args.batch_size <= args.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu


    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )
    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    align_dict = utils.load_align_dict(args.replace_unk)
    # Optimize ensemble for generation
    for model in models:
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Initialize generator
    generator = knn_build_generator(models, args, tgt_dict)
    # generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    task.load_dataset("train")
    train_set = task.datasets["train"]

    resource = {}
    resource["models"] = models
    resource["generator"] = generator
    resource["tokenizer"] = tokenizer
    resource["bpe"] = bpe
    resource["args"] = args
    resource["task"] = task
    resource["use_cuda"] = use_cuda
    resource["tgt_dict"] = tgt_dict
    resource["src_dict"] = src_dict
    resource["align_dict"] = align_dict
    resource["train_set"] = train_set
    
    return resource



def translate_using_knn_model(inputs, resource, k, lambda_, temperature):
    inputs = [inputs]
    models = resource["models"]
    generator = resource["generator"]
    tokenizer = resource["tokenizer"]
    bpe = resource["bpe"]
    args = resource["args"]
    task = resource["task"]
    use_cuda = resource["use_cuda"]
    tgt_dict = resource["tgt_dict"]
    src_dict = resource["src_dict"]
    align_dict = resource["align_dict"]
    train_set = resource["train_set"]

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    results = []
    for batch in make_batches(inputs, args, task, max_positions, encode_fn):
        bsz = batch.src_tokens.size(0)
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        constraints = batch.constraints
    if use_cuda:
        src_tokens = src_tokens.cuda()
        src_lengths = src_lengths.cuda()
        if constraints is not None:
            constraints = constraints.cuda()

        # sample, we add knn parameters in sample
        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "knn_parameter": {
                "k": k,
                "lambda": lambda_,
                "temperature": temperature,
            },
            "save_knn_informations": True,
        }
        translate_start_time = time.time()
        translations = task.inference_step(
            generator, models, sample, constraints=constraints
        )
        translate_time = time.time() - translate_start_time
        list_constraints = [[] for _ in range(bsz)]
        if args.constraints:
            list_constraints = [unpack_constraints(c) for c in constraints]
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            constraints = list_constraints[i]
            results.append(
                (
                    id,
                    src_tokens_i,
                    hypos,
                    {
                        "constraints": constraints,
                        "time": translate_time / len(translations),
                    },
                )
            )
 
    # sort output to match input order
    # for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
    #     if src_dict is not None:
    #         src_str = src_dict.string(src_tokens, args.remove_bpe)
    #         # print("S-{}\t{}".format(id_, src_str))
    #         # print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
    #         # for constraint in info["constraints"]:
    #         #     print(
    #         #         "C-{}\t{}".format(
    #         #             id_, tgt_dict.string(constraint, args.remove_bpe)
    #         #         )
    #         #     )

    # src_tokens
    src_text = [src_dict[_id] for _id in src_tokens[0]]
    print(" ".join(src_text))
    # Process top predictions
    assert len(results) == 1, "interactive mode, should have only one sentence"
    top_hypo = results[0][2][0] # store the top probability translations


    useful_results = {}
    useful_results["hypo_tokens"] = top_hypo["tokens"]
    useful_results["hypo_tokens_str"] = [tgt_dict[i] for i in top_hypo["tokens"]] 
    # only return 100 top element for simple, sort by prob
    sorted_neural_prob, sorted_neural_indices =  torch.sort(top_hypo["neural_probs"], dim=-1, descending=True)
    sorted_neural_candis = [[tgt_dict[idx] for idx in line[:100]] for line in sorted_neural_indices]
    useful_results["neural_probs"] = sorted_neural_prob[:, :100].cpu().numpy()
    useful_results["neural_candis"] = sorted_neural_candis

    sorted_combined_prob, sorted_combined_indices = torch.sort(top_hypo["combined_probs"], dim=-1, descending=True)
    sorted_combined_candis = [[tgt_dict[idx] for idx in line[:100]] for line in sorted_combined_indices]
    useful_results["combined_probs"] = sorted_combined_prob[:, :100].cpu().numpy()
    useful_results["combined_candis"] = sorted_combined_candis

    # PCA query_point and knn_neighbors
    concat_features = torch.cat((top_hypo["query_point"].view(-1, top_hypo["query_point"].shape[-1]),
        top_hypo["knn_neighbors_keys"].view(-1, top_hypo["query_point"].shape[-1])), dim=0)
    pca = PCA(n_components=2)
    pca.fit(concat_features.cpu().numpy())

    keys_shape = top_hypo["knn_neighbors_keys"].shape
    useful_results["knn_neighbors_keys"] = pca.transform(top_hypo["knn_neighbors_keys"].view(-1, keys_shape[-1]).cpu().numpy()).reshape(*keys_shape[:-1],2)
    useful_results["knn_neighbors_values"] = [[tgt_dict[idx] for idx in line[:100]] for line in top_hypo["knn_neighbors_values"].cpu().numpy()]
    useful_results["query_point"] = pca.transform(top_hypo["query_point"].cpu().numpy())
    # faiss l2 distance, not accurate
    useful_results["knn_l2_distance"] = top_hypo["knn_l2_distance"].cpu().numpy()
    # optional: recompute the l2 distance
    # useful_results["knn_l2_distance"] = torch.cdist(top_hypo["query_point"].unsqueeze(1), 
    #             top_hypo["knn_neighbors_keys"], p=2.0).squeeze(1).cpu().numpy()
    
    # sentence id and token postions
    sentence_ids = top_hypo["knn_sentence_ids"]
    token_positions = top_hypo["knn_token_positions"]

    useful_results["knn_token_positions"] = top_hypo["knn_token_positions"]
    useful_results["knn_context_src"] = []
    useful_results["knn_context_ref"] = []

    for near_neighbor_token_sent_id in sentence_ids.cpu():
        useful_results["knn_context_src"].append([])
        useful_results["knn_context_ref"].append([])
        for sent_id in near_neighbor_token_sent_id:
            sentence = train_set[sent_id.item()]
            useful_results["knn_context_src"][-1].append("  ".join([src_dict[i] for i in sentence["source"]]))
            useful_results["knn_context_ref"][-1].append("  ".join([tgt_dict[i] for i in sentence["target"]]))

    return useful_results


def make_batches(lines, args, task, max_positions, encode_fn):
    Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
    Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")
    def encode_fn_target(x):
        return encode_fn(x)


    if args.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    if args.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )

if __name__ == "__main__":
    import os
    os.environ["MODE"] = ""
    get_spatial_distribution("/data1/zhaoqf/0101/fairseq/datastore/vanilla/koran", "")
    
    






