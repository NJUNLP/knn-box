import os
import json
import torch.nn as nn
import torch.optimer as optim
import tqdm
import time
from multiprocessing import Pool

from torch.utils.data import Dataset, Dataloader

from knnbox.common_utils import Memmap, read_config, write_config, label_smoothed_nll_loss
from knnbox.datastore.utils import build_faiss_index, load_faiss_index

class PckDatastore:
    r"""
    implement pck datastore for pckmt
    """

    def __init__(
        self,
        path,
        key_dim = 768,
        value_dim = 1,
        key_dtype = "memmap_float16",
        value_dtype = "memmap_int",
        probs_4_gram_dtype = "memmap_float16"
        tgt_entropy_dtype = "memmap_float16"
        keys = None,
        values = None,
        ids_4_gram = None,
        probs_4_gram = None,
        tgt_entropy = None,
        has_keys = True,
        **kwargs,
    ):
        r"""
        Args:
            path(`str`):
                the directory to save datastore files
        
        """
        self.path = path
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.probs_4_gram_dtype = probs_4_gram_dtype
        self.tgt_entropy_dtype= tgt_entropy_dtype

        # create folder if not exist
        if not os.path.exists(path):
            os.mkdir(path)
        
        if keys is not None:
            self.keys = keys
        elif has_keys is True:
            if key_dtype.startswith("memmap"):
                print(key_dtype)
                key_dtype = key_dtype.split("_")[1]
                self.keys = Memmap(
                    os.path.join(path, "keys"),
                    dtype = key_dtype,
                    dim = key_dim,
                    mode = "w+",
                )
        else:
            self.keys = None
                
        if values is not None:
            self.values = values
        else:
            if value_dtype.startswith("memmap"):
                value_dtype = value_dtype.split("_")[1]
                self.values = Memmap(
                    os.path.join(path, "values"),
                    dtype = value_dtype,
                    dim = value_dim,
                    mode = "w+",
                )

        # ids of 4 gram before target token
        if ids_4_gram is not None:
            self.ids_4_gram = ids_4_gram
        else:
            self.ids_4_gram = Memmap(
                os.path.join(path, "ids_4_gram"),
                dtype = "int",
                dim = 4,
                mode = "w+",
            )
        # probs of 4 gram before target token 
        if probs_4_gram is not None:
            self.probs_4_gram = probs_4_gram
        else:
            probs_4_gram_dtype = probs_4_gram_dtype.split("_")[1]
            self.probs_4_gram = Memmap(
                os.path.join(path, "probs_4_gram"),
                dtype = probs_4_gram_dtype,
                dim = 4,
                mode = "w+",
            )
        
        # entropy of target token
        if tgt_entropy is not None:
            self.tgt_entropy = tgt_entropy
        else:
            tgt_entropy_dtype = tgt_entropy_dtype.split("_")[1]
            self.tgt_entropy = Memmap(
                os.path.join(path, "tgt_entropy"),
                dtype = tgt_entropy_dtype,
                dim = 1,
                mode = "w+",
            )
        

        self.faiss_index = None
        self.mask = None

        

    def add_key(self, key):
        self.keys.add(key)

    
    def add_value(self, value):
        self.values.add(value)
 

    def set_mask(self, mask):
        self.mask = mask

    
    def get_mask(self):
        return self.mask
    
    def add_ids_4_gram(self, ids_4_gram):
        r""" add the 4 gram ids before the target token"""
        self.ids_4_gram.add(ids_4_gram)
    
    def add_probs_4_gram(self, probs_4_gram):
        r""" add the 4 gram probs before the target token""" 
        self.probs_4_gram.add(probs_4_gram)
    
    def add_tgt_entropy(self, tgt_entropy):
        r""" add the target token probability"""
        self.tgt_entropy.add(tgt_entropy)
    
    
    @staticmethod
    def load(path, load_key=True):
        r"""
        load the datastore under the `path` folder

        Args:
            path(`str`):
                folder where the datastore files is stored
        Return:
            Datastore object(`Datastore`)
        """

        config = read_config(path)
        # load the memmap from disk
        keys_filename = os.path.join(path, "keys")
        has_keys = True
        if load_key is True and config["key_dtype"].startswith("memmap"):
            config["key_dtype"] = config["key_dtype"].split("_")[1]
            keys = Memmap(keys_filename, dim=config["key_dim"],
                    dtype=config["key_dtype"], mode="r",
                    capacity=config["capcity"])
        else:
            keys = None
            has_keys = False

        values_filename = os.path.join(path, "values")
        if config["value_dtype"].startswith("memmap"):
            config["value_dtype"] = config["value_dtype"].split("_")[1]
            values = Memmap(values_filename, dim=config["value_dim"],
                    dtype=config["value_dtype"], mode="r",
                    capacity=config["capcity"])

        ## TODO: load the probs and entropy files 
        # 创建datastore
        ds = Datastore(**config, keys=keys, values=values, has_keys=has_keys)
        return ds



    def load_faiss_index(self, move_to_gpu=True):
        r"""
        load faiss index, so you can use self.faiss_index
        """
        path = os.path.join(self.path, "keys.faiss")
        capcity = self.keys.capacity if self.keys is not None else self.values.capacity
        self.faiss_index = load_faiss_index(path,
        shape = (capcity, self.key_dim),
        n_probe = 32,
        move_to_gpu=move_to_gpu)
        


    
    def dump(self, verbose=True):
        r"""
        store the datastore files and config file to disk.
        Args:
            path(`str`):
                folder where you want to save the datastore files
        """
        self.keys.dump()
        self.values.dump()
        self.ids_4_gram.dump()
        self.probs_4_gram.dump()
        self.tgt_entropy.dump()
        config = {}
        config["path"] = self.path
        config["key_dim"] = self.key_dim
        config["value_dim"] = self.value_dim
        config["key_dtype"] = self.key_dtype
        config["value_dtype"] = self.value_dtype
        config["probs_4_gram_dtype"] = self.probs_4_gram_dtype
        config["tgt_entropy_dtype"] = self.tgt_entropy_dtype
        config["capcity"] = self.values.capacity
        write_config(self.path, config)
        if verbose:
            print("keys: %d " % self.keys.capacity)
            print("values: %d " % self.values.capacity)
            assert self.keys.capacity == self.values.capacity, "keys entries not equal to values"
            print("added %d entries" % config["capcity"])



    def build_faiss_index(self, verbose=False):
        r"""
        build faiss index for keys.
        the output file named `keys.faiss`
        """
        if not isinstance(self.keys, Memmap):
            print("[ERROR]: can only build faiss for Memmap / np.NDArray")
            os.exit(1)
        
        # build faiss
        build_faiss_index(self.keys.data, 
                    self.keys.data.shape,
                    os.path.join(self.path, "keys.faiss"),
                    verbose=verbose)


    def prune(self, n_of_4_gram, prune_style="random"):
        r""" prune the datastore """
        start = time.time()

        # ppl mask
        ## TODO: padding应该为0??
        ppl_mask = (self.ids_4_gram != 0).astype(np.float32) # padding
        r'''e.g., for a phrase 'it is a lovely dog' (which ends with 'dog'),
        we collect normalized ppls of all n-grams:
        - ppl of 'dog' = ppls[:1] / 1
        - ppl of 'lovely dog' = ppls[:2].sum() / 2 
        - ppl of 'a lovely dog' = ppls[:3].sum() / 3
        - ppl of 'is a lovely dog' = ppls[:4].sum() / 4
        - ppl of 'it is a lovely dog' = ppls[:5].sum() / 5
        '''
        n_gram_uniform_ppl = - np.log(self.probs_4_gram * ppl_mask + 1e-5)
        n_gram_uniform_ppl = np.concatenate([n_gram_uniform_ppl[:,:i+1].sum(-1,keepdims=True) / (i+1) \
            for i in range(n_gram_uniform_ppl.shape[-1])], axis=-1) # [datastore_size, 4]
        print("all n-grams ppl collected")

        # get the translation entropy of each token
        tgt_entropy = self.tgt_entropy[:, 0]
        print("tgt_entropy established.")

        # determin n for n_gram
        if 1 <= n_of_4_gram <= 4:
            # select the min ppl of all n-grams
            n_gram_uniform_ppl = np.min(n_gram_uniform_ppl, axis=-1)

            # caclulate the hash of n_gram
            linear_hash_weight = np.array([0]+[math.exp(i+1) for i in range(n_of_4_gram-1)])
            ids_n_gram_hash = (ids_4_gram[:, :n_of_4_gram] @ linear_hash_weight[:, None])[:, 0]
            ids_n_gram_hash = ids_n_gram_hash / np.power(np.log10(ids_n_gram_hash + 1.) + 1, 10)
            n_gram= ids_n_gram_hash + self.probs_4_gram[:, 0]
            del ids_n_gram_hash
        else:
            raise NotImplementedError("not implemented for  n = %d " % n_of_4_gram)
        
        table_n_gram_counter = Counter(n_gram)
        table_n_gram = list(table_n_gram_counter.keys())
        
        table_n_gram_idx_dict = {}
        for k in tqdm(table_n_gram):
            table_n_gram_idx_dict[k] = np.zeros(table_n_gram_counter[k], dtype=np.int64)

        for idx, gram in enumerate(tqdm(n_gram)):
            if table_n_gram_counter[gram] <= 0:
                continue
            table_n_gram_counter[gram] = -1
            table_n_gram_idx_dict[gram][table_n_gram_counter[gram]] = idx
        del table_n_gram_counter 
        print("%d way N-gram table dict established. " len(table_n_gram))

        print("start pruning..." % prune_style)
        table_n_gram_idx_dict_keys = list(table_n_gram_idx_dict.keys())
        thread_num = 30
        thread_width = len(table_n_gram_idx_dict) // thread_num + 1
        pool = Pool(process=thread_num)

        # split the n_gram_idx_dict to multi-thread
        results = [pool.apply_async(
            func=self._n_gram_prune_thread_inner_table_n_gram_idx_dict,
            args=(
                dict([(k, table_n_gram_idx_dict[k]) for k in \
                    table_n_gram_idx_dict_keys[i*thread_width:min((i+1)*thread_width, len(table_n_gram_idx_dict))]]),
                    prune_style,
                    minium_sample,
                    sample_rate,
                    n_gram_uniform_ppl if "ppl" in prune_style else None,
                    tgt_entropy if "entropy" in prune_style else None,
            )) for i in range(thread_num)]
        pool.close(), pool.join()
        table_n_gram_idx_dict = {}
        for res in results:
            table_n_gram_idx_dict.update(res.get())
        table_n_gram_idx_dict_keys = list(table_n_gram_idx_dict.keys())
        print("Pruning threads done.")


        # ---- collect pruned result ---
        print("start collecting pruned n-grams")
        thread_num = 30
        pool = Pool(process=thread_num)
        thread_width = len(table_n_gram_idx_dict) // thread_num + 1

        results = [pool.apply_async(
            func=self._collect_pruned_n_grams_thread,
            args=(
                dict([(k, table_n_gram_idx_dict[k]) \
                    for k in table_n_gram_idx_dict_keys[i*thread_width: min((i+1)*thread_width, len(table_n_gram_idx_dict))]]),                    
            ) for i in range(thread_num)
        )]
        pool.close(), pool.join()
        print("collect threads done")

        val_list, dbidx_list, key_list = [], [], []
        for res in tqdm.tqdm(results):
            val_l, dbidx_l, _, _ = res.get()
            val_list.extend(val_l)
            dbidx_list.extend(dbidx_l)
            key_list.extend([[self.keys[k] for k in np_idxs] for np_idxs in dbidx_l])
        
        print("clustering done, pruned %f of datastore, getting %d n-gram, %d nodes from %d nodes" % 
            sum([len(keys) for keys in key_list])/self.keys.end,
            len(key_list), sum([len(keys) for keys in key_list]), self.keys.end
        )
        ## TODO: 置换


    def _n_gram_prune_thread_inner_table_n_gram_idx_dict(
        self,
        table_n_gram_idx_dict: Dict,
        prune_style: str,
        mininum_sample: int,
        sample_rate: float,
        n_gram_uniform_ppl = None,
        tgt_entropy = None,
    ):
        for n_gram_str_symbol, np_idxs in tqdm(table_n_gram_idx_dict.items()):
        # for n_gram_str_symbol in tqdm(table_n_gram_idx_dict_keys):
            # np_idxs = table_n_gram_idx_dict[n_gram_str_symbol]

            selected_num = max(mininum_sample, int(sample_rate * np_idxs.shape[0]))

            # --- too sparse, we do not prune it
            if np_idxs.shape[0] <= selected_num:
                continue

            # --- 1. random selection
            if prune_style == 'random':
                table_n_gram_idx_dict[n_gram_str_symbol] = random_sample(np_idxs, selected_num)

            # --- 2. ppl pruning
            elif 'ppl' in prune_style:
                ppl_group = n_gram_uniform_ppl[np_idxs]

                if prune_style == 'prune_high_ppl':
                    # --- get lower ppl
                    mask = np.argpartition(ppl_group, selected_num)[:selected_num] 
                elif prune_style == 'prune_low_ppl':
                    # --- get higher ppl
                    mask = np.argpartition(ppl_group, -selected_num)[-selected_num:] 
                elif prune_style == 'prune_half_low_half_high_ppl':
                    # --- get half higher and half lower ppl
                    mask1 = np.argpartition(ppl_group, selected_num // 2)[:selected_num // 2] # half lower ppl
                    mask2 = np.argpartition(ppl_group, -selected_num // 2)[-selected_num // 2:] # half higher ppl
                    mask  = np.concatenate((mask1, mask2), axis=0)
                elif prune_style == 'prune_similar_ppl':
                    # --- get similar-ppl pruned
                    mask = self._ppl_split_and_sample(ppl_group, sample_rate=sample_rate)
                table_n_gram_idx_dict[n_gram_str_symbol] = np_idxs[mask]

            # --- 3. entropy pruning
            elif 'entropy' in prune_style:
                entropy_group = tgt_entropy[np_idxs]
                if prune_style == 'prune_high_entropy':
                    # --- get lower entropy
                    mask = np.argpartition(entropy_group, selected_num)[:selected_num] 
                elif prune_style == 'prune_low_entropy':
                    # --- get higher entropy
                    mask = np.argpartition(entropy_group, -selected_num)[-selected_num:] 
                elif prune_style == 'prune_half_low_half_high_entropy':
                    # --- get half higher and half lower entropy
                    mask1 = np.argpartition(entropy_group, selected_num // 2)[:selected_num // 2] # half lower entropy
                    mask2 = np.argpartition(entropy_group, -selected_num // 2)[-selected_num // 2:] # half higher entropy
                    mask  = np.concatenate((mask1, mask2), axis=0)
                elif prune_style == 'prune_similar_entropy':
                    # --- get similar-entropy pruned
                    mask = ppl_split_and_sample(entropy_group, sample_rate=sample_rate)
                table_n_gram_idx_dict[n_gram_str_symbol] = np_idxs[mask]

            # --- 4. TODO length count pruning
            else:
                raise NotImplementedError('not implemented prune_style = %s' % prune_style)
        return table_n_gram_idx_dict
    
    def _ppl_split_and_sample(
        self,
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
                clusters = random_sample(clusters, sample_nums)
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
                clusters = random_sample(clusters, sample_nums)
                # clusters = middle_k_idx(clusters, ppl_group[clusters], k=sample_nums)
                ppl_idx_clusters[i] = clusters

            mask = np.hstack(ppl_idx_clusters)
            assert mask.shape[0] <= ppl_group.shape[0], (ppl_idx_clusters)
            return mask

    def _collect_pruned_n_grams_thread(table_n_gram_idx_dict: Dict):
        len_d = len(table_n_gram_idx_dict)
        val_list = [[] for _ in range(len_d)]
        dbidx_list = [[] for _ in range(len_d)]
        for i, (n_gram_str_symbol, np_idxs) in enumerate(table_n_gram_idx_dict.items()):
            np_idxs = table_n_gram_idx_dict[n_gram_str_symbol]

            # --- slow 
            # '30-23-40' -> [30, 23, 40], the first element is the final token vocab id of this phrase
            # vocab_id = int(n_gram_str_symbol.split('.')[0])

            # --- fast solution
            # '30.0557223434982' -> the integer part is the final token vocab id of this phrase
            vocab_id = int(n_gram_str_symbol)


            val_list[i] = [vocab_id] * np_idxs.shape[0]
            dbidx_list[i] = np_idxs.tolist()
            # tgt_lens_list[i] = general_tgt_lens[np_idxs].tolist()
            # src_lens_list[i] = general_src_lens[np_idxs].tolist()
        return val_list, dbidx_list, None, None #tgt_lens_list, src_lens_list


    def dimension_reduction(self, output_dim=64, batch_size=1024):
        r""" reduct the dimesion of datastore using trained reduction network"""
        
        start_idx = 0
        # TODO: 创建reducted values memmap
        reducted_vals = Memmep(...)
        # TODO: 将压缩后的values写入到memmap
        while start_idx < self.keys.end:
            original_val = self.keys[start_idx:min(start_idx+batch_size, self.keys.end)]
            reducted_val = self.reduction_network.reduction_layer(original_val)
            reducted_vals[start_idx:min(start_idx+batch, self.keys.end)] = reducted_val.cpu().numpy()
        
    
    def train_reduction_network(
            self, 
            dr_loss_ratio = 0.0,
            nce_loss_ratio = 1.0,
            wp_loss_ratio = 0.0,
            batch_size = 1024,
            epoch = 10,
            lr = 3e-4,
            device = "cuda:0",
        ):
        r""" 
        A simple function to train reduction network """

        print("Prepare Dataset...")
        # TODO: fill parameters
        dataset = TripletDatastoreSamplingDataset1(...)
        dataloader = Dataloader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 2,
            drop_last = True,
        )
        
        print("Reduction Network Training Start...")
        self.reduction_network.to(device)
        optimizer = optim.Adam(self.reduction_network.parameters(), lr)
        for e in range(epoch):
            print("Epoch " + str(e) + " / " + str(epoch) +">>>")
            losses = []
            for data in tqdm.tqdm(dataloader):
                optimizer.zero_grad()
                loss = self.reduction_network(data, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, device)
                losses.append(loss.data)
                loss.backward()
                optimizer.step()
            print("<<< average loss: " + str(sum(losses)/len(losses)))
        print("Reduction Network Training Finished.")
        
    

class ReductionNetwork(nn.Module):
    r""" network to reduct dimension """ 
    def __init__(self, dictionary_len, input_dim, output_dim, dropout = 0.0, train_mode = False):
        self.dictionary_len = dictionary_len
        self.reduction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim // 4, output_dim),
        )
        nn.init.xavier_normal_(reduction_layer[0].weight, gain=0.01)
        nn.init.xavier_normal_(reduction_layer[-1].weight, gain=0.1)
 
        if train_mode:
            self.word_predict_layer = nn.Linear(output_dim, len(dictionary), bias=False)
            nn.init.normal_(self.word_predict_layer.weight, mean=0, std=output_dim**-0.5)

            
    def forward(self, data, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, device="cuda:0"):
        r""" forward data to get loss

        The final loss is:
            loss = dr_loss_ratio*dr_loss + nce_loss_ratio*nce_loss + wp_loss_ratio*wp_loss
        """
        assert dr_loss_ratio + nce_loss_ratio + wp_loss_ratio == 1.0, "ERROR: loss ratio's sum must equal to 1.0"
        batch_size = pivot_ids.shape[0];
        pivot_samples = data["pivot_samples"]
        positive_samples = data["positive_samples"]
        negative_samples = data["negative_samples"]
        pivot_ids = data["pivot_ids"]
        positive_ids = data["positive_ids"]
        negative_ids = data["negative_ids"]

        stack_data = torch.cat([pivot_samples, positive_samples, negative_samples], dim=0).to(device)
        stack_ids = torch.cat([pivot_ids, positive_ids, negative_ids], dim=0).to(device)

        reducted_data = self.reduction_layer(stack_data)
        reducted_pivot_data, reducted_positive_data, reducted_negative_data = \
            reducted_data[:batch_size], reducted_data[batch_size:2*batch_size], reducted_data[2*batch_size:3*batch_size]

        # I. distance ranking loss
        if dr_loss_ratio != 0.0: 
            pos_dis = nn.MSELoss(reduce=False)(reducted_pivot_data, reducted_positive_data).sum(-1)
            # here we use hingle loss instead of MSE loss to get distance of pivot between negative data
            def hingle_loss(pivot_data, negative_data, margin=10.0):
                neg_dis = nn.MSE(reduce=False)(pivot_data, negative_data).sum(-1)
                neg_dis = float()
                return neg_dis
            neg_dis = hingle_loss(reducted_pivot_data, reducted_negative_data)
            # compute weighted ranking loss
            soft_pos = 1.0 # we simply set pos_ratio = neg_ratio = 1
            soft_neg = 1.0
            soft_pos_loss = soft_pos * pos_dis;
            soft_neg_loss = soft_neg * (margin/(neg_dis + 1e-3))
            ## TODO: 这里要验证下是否除以了batch_size
            dr_loss = (soft_pos_loss + soft_neg_loss).mean()
        
        # II. noise contrasive loss
        if nce_loss_ratio != 0.0:
           nce_distance_pos = - (reducted_pos_data[:, None, :] * reducted_pivot_data[None, :, :]).sum(-1) # bsz, bsz
           nce_distance = nce_distance_pos
            r'''
            NOTE the simplest nce is to optimize among positive pairs in a batch, but sampling of positive
            pairs ignore tokens of low frequence Which make the optimization only done for high-frequence vocab.
            To address this, we optimize positive pairs nce loss along with negative pairs
            '''
            ## TODO: 这里的逻辑不是很懂
            nce_distance_pos = - (reducted_negative_data[:, None, :] * reducted_pivot_data[None, :, :]).sum(-1) # bsz, bsz
            nce_distance = torch.cat([nce_distance_pos, nce_distance_neg], axis=1)

            nce_lprobs = torch.nn.functional.log_softmax(-nce_distance, dim=-1)
            nce_target = torch.arange(end=batch_size).to(device)
            nce_loss, _ = label_smoothed_nll_loss(nce_lprobs, nce_target, reduce=True)
            nce_loss = nce_loss / float(batch_size)


        # III. word prediction loss
        if wp_loss_ratio != 0.0:
            logits = self.word_predict_layer(reducted_data)
            word_probs = nn.functional.log_softmax(logits, dim=-1)
            word_predict_loss, _ = label_smoothed_nll_loss(word_probs, stack_ids, reduce=True)
            wp_loss = word_predict_loss / float(batch_size)
        
        loss = dr_loss_ratio * dr_loss + nce_loss_ratio * nce_loss + wp_loss_ratio * wp_loss; 
        return loss



class TripletDatastoreSamplingDataset1(Dataset):
    r"""
    this dataset is used for contrastive learning sample"""

    def __init__(self,
            args,
            dictionary = None,
            use_cluster : bool = False,
            cluster_type : str = 'dbscan',
            verbose = False,
            knn_datastore = None,
            sample_rate: float = 0.4
        ):
        assert knn_datastore is not None
        assert sample_rate <= 1.0

        db_keys, db_vals = knn_datastore.keys, knn_datastore.vals

        if sample_rate != 1.0:
            random_sample = np.random.choice(np.arange(db_vals.shape[0]), size=sample_rate * db_vals.shape[0], replace=False)
            db_keys = db_keys[random_sample]
            db_vals = db_vals[random_sample]

        vocab_freq = [0 for _ in range(len(dictionary))]
        key_list = [[] for _ in range(len(dictionary))]

        # ## (deprecated) filter vocabulary from unrelevant languages
        # import langid
        # unwanted_language = ['zh', 'ko', 'ja']
        # wanted_vocab = [True for _ in range(len(dictionary))]
        # if verbose: print('unwanted vocabularies are = ')
        # for i, voc in enumerate(dictionary.symbols):
        #     voc = voc.replace('@', '') # remove bpe symbols
        #     if langid.classify(voc)[0] in unwanted_language:
        #         if verbose: print(voc, end='')
        #         wanted_vocab[i] = False
        # if verbose:
        #     print('\n total number of dictionary = %d' % len(dictionary))
        #     print("the number of unwanted vocabularies = %d, almost %f of all" % \
        #           (len(dictionary) - sum(wanted_vocab), (len(dictionary) - sum(wanted_vocab)) / len(dictionary)) )

        ## frequence collection
        for i in tqdm(range(args.dstore_size)):
            val = db_vals[i]
            vocab_freq[val] += 1
            key_list[val].append(db_keys[i])

        del db_vals
        del db_keys
        del knn_datastore

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
            
            print('start clustering ...')
            new_key_list = []
            new_val_list = []
            base_number = min_samples
            # Limited by memory, 100000 koran/it/medical (<=10M) 20000 for law/subtitles (>=19M). 
            sample_bound = 100000
            for vocab_id, keys in tqdm(enumerate(key_list)):
                if len(keys) == 0:
                    continue

                if vocab_id % 2000 == 0:
                    print('clustering %d' % vocab_id)

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

            print('we get %d clusters' % len(self.key_list))

            # # post-processing
            # for i in range(len(self.key_list)):
            #     if len(self.key_list[i]) == 0:
            #         continue
            #     self.key_list[i] = np.array(self.key_list[i])

            print('cluster done. Get %d nodes' % sum([len(keys) for keys in self.key_list]))

        ## (deprecated)
        # if verbose: print('==== wanted vocabularies\' frequence ====')
        # with open('val_list_it', 'w') as f:
        #     for d, v, wv in zip(dictionary.symbols, vocab_freq, wanted_vocab):
        #         f.write(str(v) + '\n')
        #         if wv and verbose:
        #             print(d, v, end='  ')

        ## statistics collection of vocab frequency
        self.larger_than_2_vocab  = [i for i, v in enumerate(self.key_list) if len(v) >= 2 ]
        self.larger_than_1_vocab  = [i for i, v in enumerate(self.key_list) if len(v) >= 1 ]
        assert len(self.larger_than_2_vocab) > 0, 'the datastore is too sparse to conduct a good baseline'

        ## add up the cluster centroid into the cluster
        for i, keys in enumerate(self.key_list):
            if len(keys) > 0:
                self.key_list[i].append(torch.tensor(keys).float().mean(dim=0).half().numpy())
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
            'negative_samples': torch.tensor(negative_sample),
            'negative_ids': negative_vocab,
            'positive_samples': torch.tensor(positive_sample),
            'positive_ids': idx,
            'pivot_samples': torch.tensor(pivot_sample),
            'pivot_ids': idx,
        }
        return batch_dict

    def __len__(self):
        return len(self.larger_than_2_vocab)
