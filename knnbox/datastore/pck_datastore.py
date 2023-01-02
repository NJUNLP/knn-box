import time
import math
import os
import json
from random import randint, sample

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import Birch, DBSCAN
from multiprocessing import Pool
from collections import Counter
from tqdm import tqdm

from knnbox.datastore import Datastore
from knnbox.common_utils import (
    Memmap,
    write_config,
    read_config,
    label_smoothed_nll_loss,
)


class PckDatastore(Datastore, nn.Module):
    """ Implementation of PCK-MT datastore """ 
    def __init__(
        self,
        path,
        dictionary_len,
        reduction_network_input_dim = None,
        reduction_network_output_dim = None,
        datas=None,
        **kwargs,
        ):
        Datastore.__init__(self, path, datas, **kwargs)
        nn.Module.__init__(self)      
        self.dictionary_len = dictionary_len

        # create a network for dimension reduction when need
        if reduction_network_input_dim and reduction_network_output_dim and dictionary_len:
            self.reduction_network = ReductionNetwork(
                    dictionary_len, 
                    reduction_network_input_dim,
                    reduction_network_output_dim,
                    train_mode=False,
            ) 
            self.reduction_network_input_dim = reduction_network_input_dim
            self.reduction_network_output_dim = reduction_network_output_dim


    def prune_size(
        self,
        output_path,
        n_of_4_gram = 4, # we save 4 gram info, but we can use less gram to prune
        prune_style = "random",
        sample_rate = 0.1, 
        minimum_sample = 2, 
        thread_num = 30,
    ):
        r""" prune the datastore size """
        start_time = time.time()

        # ppl mask
        ppl_mask = (self.datas["ids_4_gram"].data != 0).astype(np.float32) # padding
        r'''e.g., for a phrase 'it is a lovely dog' (which ends with 'dog'),
        we collect normalized ppls of all n-grams:
        - ppl of 'dog' = ppls[:1] / 1
        - ppl of 'lovely dog' = ppls[:2].sum() / 2 
        - ppl of 'a lovely dog' = ppls[:3].sum() / 3
        - ppl of 'is a lovely dog' = ppls[:4].sum() / 4
        - ppl of 'it is a lovely dog' = ppls[:5].sum() / 5
        '''
        n_gram_uniform_ppl = - np.log(self["probs_4_gram"].data * ppl_mask + 1e-5)
        n_gram_uniform_ppl = np.concatenate([n_gram_uniform_ppl[:,:i+1].sum(-1, keepdims=True) / (i+1)
            for i in range(n_gram_uniform_ppl.shape[-1])], axis=-1)
        print("[prune size] all n-grams ppl collected")

        # get the translation entropy of each token
        tgt_entropy = self["entropy"].data

        # determin n for n_gram
        if 1 <= n_of_4_gram <= 4:
            # select the min ppl of all n-grams
            n_gram_uniform_ppl = np.min(n_gram_uniform_ppl,  axis=-1)
            # calculate the hash of n_gram
            linear_hash_weight = np.array([0]+[math.exp(i+1) for i in range(n_of_4_gram-1)])
            ids_n_gram_hash = (self["ids_4_gram"].data[:, :n_of_4_gram] @ linear_hash_weight[:, None])[:, 0]
            ids_n_gram_hash = ids_n_gram_hash / np.power(np.log10(ids_n_gram_hash + 1.) + 1, 10)
            ids_n_gram_hash = ids_n_gram_hash
            n_gram = ids_n_gram_hash + self["ids_4_gram"].data[:, 0]
            del ids_n_gram_hash
            # slow solution
            # n_gram = [".".join([str(w) for w in grm_n]) for grm_n in self["ids_4_gram"].data[:, :n_of_4_gram]]
        else:
            raise NotImplementedError("not implemented for n = %d" % n_of_4_gram)
        
        table_n_gram_counter = Counter(n_gram)
        table_n_gram = list(table_n_gram_counter.keys())

        table_n_gram_idx_dict = {}
        for k in table_n_gram:
            table_n_gram_idx_dict[k] = np.zeros(table_n_gram_counter[k], dtype=np.int64)

        # put the idx to table_n_gram_idx_dict, the key is hash of n_gram 
        for idx, gram in enumerate(n_gram):
            if table_n_gram_counter[gram] <= 0:
                continue
            table_n_gram_counter[gram] -= 1
            table_n_gram_idx_dict[gram][table_n_gram_counter[gram]] = idx
        del table_n_gram_counter
        print("[prune size] %d way N-gram table dict established. " % len(table_n_gram))

        r"""
        NOTE: about table_n_gram_idx_dict
        For a trainset that contains 6 sentences:
            I:   'this is a good place'
            II:  'it is rainy.'
            III: 'he is good'
            IV:  'i think he is excellent'
            V:   'yes he is'
            VI:  'is it ?'
        We build the datastore:
        0-this, 1-is, 2-a, 3-good, 4-place,
        5-it, 6-is, 7-rainy,
        8-he, 9-is, 10-good,
        11-i, 12-think, 13-he, 14-is, 15-excellent,
        16-yes, 17-he, 18-is,
        19-is, 20-it, 21-?
        the 1-gram list of "is":  [
            [1('this is')],
            [6('it is')],
            [9('he is')],
            [14('he is')],
            [18('he is')],
            [19('is')]
        ]
        the 2-gram list of "is" that ends with the token "is": [
            [1 ('this is')],
            [6 ('it is')],
            [9, 14, 18 ('he is')],
            [19 ('[padding] is')]
        ]
        etc.
        """

        # start pruning 
        print("[prune size] start %s pruning ..." % prune_style)

        thread_width = len(table_n_gram_idx_dict) // thread_num + 1
        pool = Pool(processes=thread_num)

        table_n_gram_idx_dict_keys = list(table_n_gram_idx_dict.keys())

        # multi thread pruning
        results = [pool.apply_async(
            func = self._n_gram_prune_thread_inner_table_n_gram_idx_dict,
            args = (
                dict([(k, table_n_gram_idx_dict[k]) for k in \
                    table_n_gram_idx_dict_keys[i*thread_width:min((i+1)*thread_width, 
                    len(table_n_gram_idx_dict))]]),
                prune_style,
                minimum_sample,
                sample_rate,
                n_gram_uniform_ppl if "ppl" in prune_style else None,
                tgt_entropy if "entropy" in  prune_style else None,
            ),
        ) for i in range(thread_num)]
        pool.close()
        pool.join()

        # union the result
        table_n_gram_idx_dict = {}
        for res in results: 
            table_n_gram_idx_dict.update(res.get())
        table_n_gram_idx_dict_keys = list(table_n_gram_idx_dict.keys())
        pool = Pool(processes=thread_num)
        thread_width = len(table_n_gram_idx_dict) // thread_num + 1
        print("[prune size] start collect result...")
        results = [pool.apply_async(
           func = self._collect_pruned_n_grams_thread,
           args = (
                dict([(k, table_n_gram_idx_dict[k]) \
                    for k in table_n_gram_idx_dict_keys[i*thread_width:min((i+1)*thread_width, 
                    len(table_n_gram_idx_dict))]]),
           ),
        ) for i in range(thread_num)]

        pool.close()
        pool.join()

        output_datastore = Datastore(path=output_path)
        for res in results:
            vals_l, dbidx_l, tgt_lens_l, src_lens_l = res.get()
            vals_l = [val for vals in vals_l for val in vals]
            keys_l = [self["keys"].data[dbidx] for dbidxs in dbidx_l for dbidx in dbidxs]
            vals = np.array(vals_l, dtype=self["vals"].data.dtype)
            keys = np.array(keys_l, dtype=self["keys"].data.dtype) 

            output_datastore["keys"].add(keys)
            output_datastore["vals"].add(vals)
        output_datastore.dump()
        output_datastore.build_faiss_index("keys")
        print("Prune Finshed: origined size->%d pruned size->%d" % 
                (tgt_entropy.shape[0], output_datastore["keys"].shape[0]))

    @classmethod
    def random_sample(cls, keys, nums):
        r"""random sample if keys' size bigger than nums """
        assert type(keys) in [list, np.ndarray], type(keys)
        if isinstance(keys, list):
            if len(keys) > nums:
                return random.sample(keys, nums)
            else:
                return keys
        else:
            if keys.shape[0] > nums:
                return keys[np.random.choice(keys.shape[0], nums, replace=False)]
            else:
                return keys


    @classmethod
    def _n_gram_prune_thread_inner_table_n_gram_idx_dict(
        cls,
        table_n_gram_idx_dict,
        prune_style,
        minimum_sample,
        sample_rate,
        n_gram_uniform_ppl = None,
        tgt_entropy = None,
    ):
        r"""prune the items which has same n-gram hash code.
            the prune policy has: random, ppl, tgt_entropy
        """
        for n_gram_str_symbol, np_idxs in table_n_gram_idx_dict.items():
            
            selected_num = max(minimum_sample, int(sample_rate*np_idxs.shape[0]))
            # -- to sparse, dont prune it
            if np_idxs.shape[0] <= selected_num:
                continue

            # -- 1. random selection
            if prune_style == "random":
                table_n_gram_idx_dict[n_gram_str_symbol] = cls.random_sample(np_idxs, selected_num)
            # -- 2. ppl pruning
            elif "ppl" in prune_style:
                ppl_group = n_gram_uniform_ppl[np_idxs]

                if prune_style == "prune_high_ppl":
                    mask = np.argpartition(ppl_group, selected_num)[:selected_num]
                elif prune_style == "prune_low_ppl":
                    mask = np.argpartition(ppl_group, - selected_num)[-selected_num:]
                elif prune_style == "prune_half_low_half_high_ppl":
                    mask1 = np.argpartition(ppl_group, selected_num // 2)[:selected_num // 2] # half lower ppl
                    mask2 = np.argpartition(ppl_group, -selected_num // 2)[-selected_num // 2:] # half higher ppl
                    mask  = np.concatenate((mask1, mask2), axis=0)
                elif prune_style == "prune_similar_ppl":
                    # use similar ppl prune 
                    mask = cls.ppl_split_and_sample(ppl_group, sample_rate=sample_rate)
                # select 
                table_n_gram_idx_dict[n_gram_str_symbol] = np_idxs[mask]
            
            # -- 3. entropy pruning
            elif "entropy" in prune_style:
                entropy_group = tgt_entropy[np_idxs]
                if prune_style == "prune_high_entropy":
                    # -- get lower entropy
                    mask = np.argpartition(entropy_group, selected_num)[:selected_num]
                elif prune_style == "prune_low_entropy":
                    # -- get higher entropy
                    mask = np.argpartition(entropy_group, - selected_num)[-selected_num:]
                elif prune_style == "prune_half_low_half_high_entropy":
                     # --- get half higher and half lower entropy
                    mask1 = np.argpartition(entropy_group, selected_num // 2)[:selected_num // 2] # half lower entropy
                    mask2 = np.argpartition(entropy_group, -selected_num // 2)[-selected_num // 2:] # half higher entropy
                    mask  = np.concatenate((mask1, mask2), axis=0)
                elif prune_style == 'prune_similar_entropy':
                    # --- get similar-entropy pruned
                    mask = cls.ppl_split_and_sample(entropy_group, sample_rate=sample_rate)
                table_n_gram_idx_dict[n_gram_str_symbol] = np_idxs[mask]
            
            # -- 4. TODO length count pruning
            else:
                raise NotImplementedError("not implemented prune_style = %s" % prune_style)
        
        return table_n_gram_idx_dict

    @classmethod
    def ppl_split_and_sample(
        cls, 
        ppl_group: np.array,
        sample_rate: float = 0.3,
        translation_cost_threshold : float = 1.5,
        minimum_sample: int = 2
        ):
        if ppl_group.shape[0] > 1e4:
            # linear cluster (faster, not optical but acceptable)
            sc = Birch(n_clusters=None, threshold=translation_cost_threshold)#, branching_factor=256)
            clustering = sc.fit(ppl_group[:, None]) # train
            labels = clustering.labels_

            ppl_clusters = [[] for _ in range(labels.max() + 1)]
            for n in range(labels.shape[0]):
                if labels[n] == -1: ## isolated node
                    continue
                ppl_clusters[labels[n]].append(n)
            for i, clusters in enumerate(ppl_clusters):
                clusters = np.array(clusters)
                sample_nums = max(min(minimum_sample, clusters.shape[0]), int(sample_rate * clusters.shape[0]))
                clusters = cls.random_sample(clusters, sample_nums)
                # clusters = middle_k_idx(clusters, ppl_group[clusters], k=sample_nums)
                ppl_clusters[i] = clusters
                
            for n in range(labels.shape[0]):
                if labels[n] == -1: ## isolated node
                    ppl_clusters.append(np.array([n], dtype=np.int))
            ppl_clusters = [ppl_index for ppl_index in ppl_clusters if ppl_index.shape[0] > 0]
            mask = np.hstack(ppl_clusters)
            assert mask.shape[0] <= ppl_group.shape[0]
            return mask
        else:
            # affinity greedy searching
            ppl_affinity = ppl_group[None] - ppl_group[:, None]
            ppl_similar = np.abs(ppl_affinity) <= translation_cost_threshold
            ppl_idx_clusters = []

            idx_empty = np.arange(ppl_similar.shape[0])
            while ppl_similar.sum() != 0.:
                ppl_similar_numbers = ppl_similar.astype(np.float32).sum(-1)
                ppl_max_similar_idx = np.argmax(ppl_similar_numbers)
                select_mask = ppl_similar[ppl_max_similar_idx]
                ppl_idx_clusters.append(idx_empty[select_mask])
                ppl_similar = ppl_similar[~select_mask]
                ppl_similar = ppl_similar[:, ~select_mask]
                idx_empty = idx_empty[~select_mask]

            for i, clusters in enumerate(ppl_idx_clusters):
                sample_nums = max(min(minimum_sample, clusters.shape[0]), int(sample_rate * clusters.shape[0]))
                clusters = cls.random_sample(clusters, sample_nums)
                # clusters = middle_k_idx(clusters, ppl_group[clusters], k=sample_nums)
                ppl_idx_clusters[i] = clusters

            mask = np.hstack(ppl_idx_clusters)
            assert mask.shape[0] <= ppl_group.shape[0], (ppl_idx_clusters)
            return mask
        

    @classmethod
    def _collect_pruned_n_grams_thread(cls, table_n_gram_idx_dict):
        r""" 
        for a dict {"3.12":[1,4,3], "4.52":[6,9], "3.89":[11,2]}
        we return: 
            val_list: [[3,3,3],[4,4],[3,3]]
            dbidx_list: [[1,4,3],[6,9],[11,2]]
        """
        len_d = len(table_n_gram_idx_dict)
        val_list = [[] for _ in range(len_d)]
        dbidx_list = [[] for _ in range(len_d)]
        for i, (n_gram_str_symbol, np_idxs) in enumerate(table_n_gram_idx_dict.items()):
            np_idxs = table_n_gram_idx_dict[n_gram_str_symbol]

            vocab_id = int(n_gram_str_symbol)

            val_list[i] = [vocab_id] * np_idxs.shape[0]
            dbidx_list[i] = np_idxs.tolist()
        
        return val_list, dbidx_list, None, None # tgt_lens_list, src_lens_list


    def train_reduction_network(
            self, 
            triplet_dataset,
            batch_size,
            dr_loss_ratio,
            nce_loss_ratio,
            wp_loss_ratio,
            lr,
            min_lr,
            patience,
            max_update,
            log_path,
            valid_interval,
            device = "cuda:0",
            ):
        r""" a simple function to train reduction network"""
        assert self.training, "Pytorch is not on trainning mode"
        assert max_update > valid_interval, "max_update must bigger than valid_interval"
        tb_writer = None # tensorboardX
        try:
            from tensorboardX import SummaryWriter
            tb_writer = SummaryWriter(log_path)
        except:
            print("[train reduction network] " 
            "tensorboardX not Installed. we won't record the log info for you!")
        dataloader = DataLoader(
            dataset = triplet_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 2,
            drop_last = False,
        )
        valid_dataloader = DataLoader(
            dataset = triplet_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 2,
            drop_last = False,
        )
        print("[train reduction network] Start Training Reduction Network...")
        self.reduction_network.to(device)
        self.reduction_network.train() # switch to train mode, enable dropout
        optimizer = optim.Adam(self.reduction_network.parameters(), lr, betas=(0.9, 0.98))
        # lerning scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode = "min",
                    patience = 5,
                    min_lr = min_lr,
                    factor = 0.5, 
                    )
        min_valid_loss = 1e7
        best_checkpoint = None
        no_improved_cnt = 0
        pbar = tqdm(total=max_update)
        update_step = 0
        valid_losses = []
        valid_cnt = 0
        break_flag = False

        while True:
            if break_flag:
                break
            for data in dataloader:
                if update_step >= max_update:
                    break_flag = True
                    break
                # validdation
                if (update_step + 1) % valid_interval == 0:
                    valid_losses = []
                    for valid_data in valid_dataloader:                
                        with torch.no_grad():
                            valid_loss = \
                                self.reduction_network(valid_data, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, device)
                            valid_losses.append(valid_loss.item())
                    avg_valid_loss = sum(valid_losses) / len(valid_losses)
                    if tb_writer:
                        tb_writer.add_scalar("valid_loss", avg_valid_loss, update_step)
                    print("valid loss after update %d steps: %f" %(update_step, avg_valid_loss))
                    # save checkpoint when get a better loss
                    if avg_valid_loss < min_valid_loss:
                        print("%f is a new best valid loss" % avg_valid_loss)
                        best_checkpoint = self.reduction_network.state_dict()
                        min_valid_loss = avg_valid_loss
                        no_improved_cnt = 0
                    else:
                        no_improved_cnt += 1
                        print("not improved for %d / %d validations." % (no_improved_cnt, patience))
                        if no_improved_cnt >= patience:
                            print("\nEarly stoped because not improved for %d validations." % no_improved_cnt)
                            break_flag = True
                            break
                    scheduler.step(avg_valid_loss)
                    update_step += 1
                    pbar.update(1)
                
                # train
                train_loss = self.reduction_network(data, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, device)
                if tb_writer:
                    tb_writer.add_scalar("train_loss", train_loss.item(), update_step)
                pbar.update(1)
                pbar.set_postfix(step=update_step, loss=train_loss.item())
                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.reduction_network.parameters(), 1.0)
                optimizer.step()
                update_step += 1

        # save the best checkpoint 
        self.reduction_network.load_state_dict(best_checkpoint)
        print("best checkpoint with valid loss %f ." % min_valid_loss)
        print("Reduction Network Training Finished.")


    def vector_reduct(self, x, device="cuda:0"):
        r""" reduct the input x with reduct network """
        self.reduction_network = self.reduction_network.to(device)
        self.reduction_network.eval()
        x = x.to(device)
        assert x.size()[-1] == self.reduction_network_input_dim, "Error: The vector size is not correct!"
        with torch.no_grad():
            reducted_x = self.reduction_network.reduction_layer(x)
        return reducted_x


    def reconstruct_keys_with_reduction_network(self, output_dir, batch_size=100):
        print("[reduct dimension] Start reduct keys' dimension using trained network")
        start_idx = 0
        key_size = self["keys"].size
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        new_keys = Memmap(os.path.join(output_dir, "keys.npy"), mode="w+")
        while start_idx < key_size:
            end_idx = min(start_idx+batch_size, key_size)
            original_key = self["keys"].data[start_idx:end_idx]
            original_key = torch.tensor(original_key, dtype=torch.float)
            reduct_key = self.vector_reduct(original_key)
            new_keys.add(reduct_key.half())
            start_idx = end_idx 
        print("[reduct dimension] Done.")
        return new_keys


    @classmethod
    def load(cls, path, load_list, load_network):
        r"""
        load the datastore from the `path` folder

        Args:
            path(`str`):
                folder where the datastore files is stored
            load_list(`list`):
                specify the data name which we want to load
        Return:
            Datastore object(`Datastore`)
        """
        
        datas = {}
        config = read_config(path)
         
        for name in load_list:
            assert name in config["data_list"], "You haven't save {} but you list it in load_list".format(name)
            if os.path.exists(os.path.join(path, name+".npy")):
                _info = config["data_infos"][name]
                datas[name] = Memmap(
                                filename=os.path.join(path, name+".npy"),
                                shape=_info["shape"],
                                dtype=_info["dtype"],
                                mode="r+",
                            )
        dictionary_len = config["dictionary_len"]
        if load_network: 
            reduction_network_input_dim = config["reduction_network_input_dim"]
            reduction_network_output_dim = config["reduction_network_output_dim"]
            pck_datastore = cls(path, 
                                dictionary_len,
                                reduction_network_input_dim,
                                reduction_network_output_dim,
                                datas,
                            )
            pck_datastore.load_state_dict(torch.load(os.path.join(path, "reduct_network.pt")), strict=False)
        else:
            pck_datastore = cls(path,
                                dictionary_len,
                                None,
                                None,
                                datas,
                            )
        return pck_datastore

    def dump(self, verbose=True, dump_list=None, dump_network=False):
        r"""
        store the datastore files and config file to disk.
        
        Args:
            verbose: whether to display detailed infomation
            dump_list: specify the data names which you want to dump. if dump_list is None, dump all data
        """

        config = {}
        config["data_list"] = []
        config["data_infos"] = {}

        for name in self.datas.keys():
            # we always dump all infomations
            config["data_list"].append(name)
            config["data_infos"][name] = {
                "name": name,
                "shape": self.datas[name].shape,
                "dtype": str(self.datas[name].dtype),
            }
            if dump_list is None or name in dump_list:
                # dump the data to disk
                self.datas[name].dump()
                if verbose:
                    print("["+name+".npy: "+str(config["data_infos"][name]["shape"])+" saved successfully ^_^ ]")
        
        # some useful info
        config["dictionary_len"] = self.dictionary_len
        if dump_network:
            config["reduction_network_input_dim"] = self.reduction_network_input_dim
            config["reduction_network_output_dim"] = self.reduction_network_output_dim
            # save checkpoint
            torch.save(self.state_dict(), os.path.join(self.path, "reduct_network.pt"))
        write_config(self.path, config)


    def set_target(self, x):
        self.tgt_ids = x
    
    def get_target(self):
        return self.tgt_ids


class ReductionNetwork(nn.Module):
    r""" network to compress dimension """
    def __init__(self, dictionary_len, input_dim, output_dim, dropout = 0.0, train_mode = True):
        super().__init__()
        self.dictionary_len = dictionary_len
        self.reduction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim // 4, output_dim),
        )
        nn.init.xavier_normal_(self.reduction_layer[0].weight, gain=0.01)
        nn.init.xavier_normal_(self.reduction_layer[-1].weight, gain=0.1)
 
        if train_mode:
            self.word_predict_layer = nn.Linear(output_dim, self.dictionary_len, bias=False)
            nn.init.normal_(self.word_predict_layer.weight, mean=0, std=output_dim**-0.5)

            
    def forward(self, data, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, device="cuda:0"):
        r""" forward data to get loss

        The final loss is:
            loss = dr_loss_ratio*dr_loss + nce_loss_ratio*nce_loss + wp_loss_ratio*wp_loss
        """
        assert dr_loss_ratio + nce_loss_ratio + wp_loss_ratio == 1.0, "ERROR: loss ratio's sum must equal to 1.0"
        pivot_samples = data["pivot_samples"]
        positive_samples = data["positive_samples"]
        negative_samples = data["negative_samples"]
        pivot_ids = data["pivot_ids"]
        positive_ids = data["positive_ids"]
        negative_ids = data["negative_ids"]
        batch_size = pivot_ids.shape[0]

        stack_data = torch.cat([pivot_samples, positive_samples, negative_samples], dim=0).to(device)
        stack_ids = torch.cat([pivot_ids, positive_ids, negative_ids], dim=0).to(device)

        reducted_data = self.reduction_layer(stack_data)
        reducted_pivot_data, reducted_positive_data, reducted_negative_data = \
            reducted_data[:batch_size], reducted_data[batch_size:2*batch_size], reducted_data[2*batch_size:3*batch_size]

        # I. distance ranking loss
        dr_loss = 0.
        if dr_loss_ratio != 0.0: 
            pos_dis = nn.MSELoss(reduce=False)(reducted_pivot_data, reducted_positive_data).sum(-1)
            # here we use hingle loss instead of MSE loss to get distance of pivot between negative data
            margin = 10.
            def hingle_loss(pivot_data, negative_data, margin):
                neg_dis = nn.MSELoss(reduce=False)(pivot_data, negative_data).sum(-1)
                neg_dis = (neg_dis < margin).float() * neg_dis + (neg_dis >= margin).float() * margin
                return neg_dis

            neg_dis = hingle_loss(reducted_pivot_data, reducted_negative_data, margin)
            # compute weighted ranking loss
            soft_pos = 1.0 # we simply set pos_ratio = neg_ratio = 1
            soft_neg = 1.0
            soft_pos_loss = soft_pos * pos_dis;
            soft_neg_loss = soft_neg * (margin/(neg_dis + 1e-3))
            dr_loss = (soft_pos_loss + soft_neg_loss).mean()
        
        # II. noise contrasive loss
        nce_loss = 0
        if nce_loss_ratio != 0.0:
            nce_distance_pos = - (reducted_positive_data[:, None, :] * reducted_pivot_data[None, :, :]).sum(-1) # bsz, bsz
            nce_distance = nce_distance_pos
            r'''
            NOTE the simplest nce is to optimize among positive pairs in a batch, but sampling of positive
            pairs ignore tokens of low frequence Which make the optimization only done for high-frequence vocab.
            To address this, we optimize positive pairs nce loss along with negative pairs
            '''
            nce_distance_pos = - (reducted_positive_data[:, None, :] * reducted_pivot_data[None, :, :]).sum(-1) # bsz, bsz
            nce_distance_neg = - (reducted_negative_data[:, None, :] * reducted_pivot_data[None, :, :]).sum(-1) # bsz, bsz
            nce_distance = torch.cat([nce_distance_pos, nce_distance_neg], axis=1)

            nce_lprobs = torch.nn.functional.log_softmax(-nce_distance, dim=-1) # the larger, the worse
            nce_target = torch.arange(end=batch_size).to(device)
            nce_loss = label_smoothed_nll_loss(nce_lprobs, nce_target, 1e-3, reduce=True)
            nce_loss = nce_loss / float(batch_size)


        # III. word prediction loss
        wp_loss = 0
        if wp_loss_ratio != 0.0:
            logits = self.word_predict_layer(reducted_data)
            word_probs = nn.functional.log_softmax(logits, dim=-1)
            word_predict_loss = label_smoothed_nll_loss(word_probs, stack_ids, 1e-3, reduce=True)
            wp_loss = word_predict_loss / float(batch_size)
        
        loss = dr_loss_ratio * dr_loss + nce_loss_ratio * nce_loss + wp_loss_ratio * wp_loss; 
        return loss



class TripletDatastoreSamplingDataset(Dataset):
    r"""
    this dataset is used for contrastive learning sample"""
    def __init__(self, 
            dictionary_len,
            use_cluster : bool = False,
            cluster_type : str = 'dbscan',
            verbose = False,
            db_keys = None,
            db_vals = None,
            sample_rate: float = 0.4,
            min_samples = 4,
        ):
        assert sample_rate <= 1.0
        
        if sample_rate != 1.0:
            random_sample = np.random.choice(np.arange(db_vals.shape[0]), 
                        size=int(sample_rate*db_vals.shape[0]), replace=False)
            sample_db_keys = db_keys[random_sample]
            sample_db_vals = db_vals[random_sample]
        else:
            sample_db_keys = db_keys
            sample_db_vals = db_vals 

        vocab_freq = [0 for _ in range(dictionary_len)]
        key_list = [[] for _ in range(dictionary_len)]

        ## frequence collection
        print("[prepare dataset] collecting frequence...")
        for i in tqdm(range(sample_db_vals.shape[0])):
            val = sample_db_vals[i]
            vocab_freq[val] += 1
            key_list[val].append(sample_db_keys[i])

        del sample_db_vals
        del sample_db_keys

        if use_cluster:
            ## inner clustering refine
            cluster_algorithm_list = ['spectrum', 'dbscan']
            assert cluster_type in cluster_algorithm_list, 'the cluster algorithm should be in the list: ' + ' '.join(cluster_algorithm_list)
            
            if cluster_type == 'spectrum':
                from sklearn.cluster import SpectralClustering
                sc = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', n_init=3, n_neighbors=min_samples)
            elif cluster_type == 'dbscan':
                from sklearn.cluster import DBSCAN
                sc = DBSCAN(eps=10, min_samples=min_samples)
            
            print('[prepare dataset] start clustering, may take long time...')
            new_key_list = []
            new_val_list = []
            base_number = min_samples
            # Limited by memory, 100000 koran/it/medical (<=10M) 20000 for law/subtitles (>=19M). 
            sample_bound = 100000
            
            pbar = tqdm(total=len(key_list))
            for idx, keys in enumerate(reversed(key_list)):
                pbar.update(1)
                # because we reversed
                vocab_id = dictionary_len - idx - 1
                if len(keys) == 0:
                    continue

                '''
                key_list[0] is a list of all-zero keys, because vocab[0] is '<s>'
                key_list[1~3] are not all-zero keys, of which the vocabs are '<pad> </s> <unk>'
                '''
                if vocab_id < 4 and vocab_id != 2:
                    continue

                if len(keys) <= base_number:
                    new_key_list.append(keys)
                    new_val_list.append([vocab_id for _ in range(len(keys))])
                    continue

                ## to decrease the computation
                if len(keys) > sample_bound:
                    keys = sample(keys, sample_bound)

                sc.n_clusters = int(math.log(len(keys)+base_number, base_number))
                sc.n_neighbors = min(len(keys), min_samples)

                keys = np.array(keys)

                clustering = sc.fit(keys)
                labels = clustering.labels_

                tmp_key = [[] for _ in range(labels.max()+1)]
                for n in range(labels.shape[0]):
                    if labels[n] == -1:
                        continue
                    tmp_key[labels[n]].append(keys[n])
                    # print(labels[j], end=' ')
                tmp_key = [key for key in tmp_key if len(key) != 0]
                new_key_list.extend(tmp_key)

                tmp_val = [[vocab_id for _ in range(len(key))] for key in tmp_key]
                new_val_list.extend(tmp_val)
                assert len(tmp_key) == len(tmp_val)

            del key_list
            self.key_list = new_key_list
            self.val_list = new_val_list
            '''
            After target-side clustering, tokens of the same vocab may be split
            into different slices of this new_val_list, like:
            [
             [5,5,5], [5,5,5,5,5],
             [6,], [6,6,6,6], [6,6,6], [6,6],
             [7],
             [8,8,8,8], [8,8],
              ...
            ]
            '''

            print('\nwe get %d clusters' % len(self.key_list))

            # # post-processing
            # for i in range(len(self.key_list)):
            #     if len(self.key_list[i]) == 0:
            #         continue
            #     self.key_list[i] = np.array(self.key_list[i])

            print('cluster done. Get %d nodes' % sum([len(keys) for keys in self.key_list]))
        
        ## statistics collection of vocab frequency
        self.larger_than_2_vocab  = [i for i, v in enumerate(self.key_list) if len(v) >= 2 ]
        self.larger_than_1_vocab  = [i for i, v in enumerate(self.key_list) if len(v) >= 1 ]
        assert len(self.larger_than_2_vocab) > 0, 'the datastore is too sparse to conduct a good baseline'

        ## add up the cluster centroid into the cluster
        for i, keys in enumerate(self.key_list):
            if len(keys) > 0:
                self.key_list[i].append(torch.tensor(np.array(keys)).float().mean(dim=0).half().numpy())
                self.val_list[i].append(self.val_list[i][0])

    def __getitem__(self, idx):
        idx = idx % len(self.larger_than_2_vocab)
        pivot_sample = self.key_list[idx][-1]
        positive_sample = sample(self.key_list[idx][:-1], 1)[0]
 
        while True:
            idx_neg = sample(self.larger_than_1_vocab, 1)[0]
            if idx_neg != idx:
                break

        idx_neg_subidx = sample(range(len(self.key_list[idx_neg])), 1)[0]
        negative_sample = self.key_list[idx_neg][idx_neg_subidx]
        negative_vocab = self.val_list[idx_neg][idx_neg_subidx]

        batch_dict = {
            'negative_samples': torch.tensor(negative_sample, dtype=torch.float),
            'negative_ids': negative_vocab,
            'positive_samples': torch.tensor(positive_sample, dtype=torch.float),
            'positive_ids': idx,
            'pivot_samples': torch.tensor(pivot_sample, dtype=torch.float),
            'pivot_ids': idx,
        }
        return batch_dict

    def __len__(self):
        return len(self.larger_than_2_vocab)
