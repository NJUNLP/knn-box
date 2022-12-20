import os
import json
from knnbox.common_utils import Memmap, read_config, write_config
from knnbox.datastore.utils import build_faiss_index, load_faiss_index 


class Datastore:
    r"""
    implement vanilla datastore
    """

    def __init__(
        self,
        path,
        datas=None,
        **kwargs,
    ):
        r"""
        Args:
            path(`str`):
                the directory to save datastore files
            datas(`dict`):
                the dict of inner data
            data_infos(`dict`):
                The infomations of datastore inner data
        
        """
        self.path = path
        # initialize datas
        self.datas = datas if datas is not None else {}
        # create folder if not exist
        if not os.path.exists(path):
            os.makedirs(path)
    

    def __getitem__(self, name):
        r""" access  inner data
        Usage:
            ds = Datastore(path="/home/datastore")
            a = torch.rand(3,1024)
            ds["keys"].add(a)
            b = torch.rand(3,1)
            ds["vals"].add(b)
        """
        if name not in self.datas:
            # Create if no exists
            self.datas[name] = Memmap(filename=os.path.join(self.path, name+".npy"), mode="w+")
        return self.datas[name]


    def __setitem__(self, name, data):
        r""" set inner data directory
        Usage:
            ds = Datastore(path="/home/datastore")
            mp = Memmap("/home/vals.npy", mode="r")
            ds["vals"] = mp
        """
        assert isinstance(data, Memmap), "__setitme__ is designed for set Memmap object"
        self.datas[name] = data

    def __delitem__(self, name):
        r""" delete a inner data """
        if name in self.datas:
            del self.datas[name]
    

    def set_pad_mask(self, mask):
        r""" 
        save the pad mask 
        """ 
        self.mask = mask


    def get_pad_mask(self):
        r"""
        get the saved mask
        """
        assert hasattr(self, "mask"), "You should set pad mask first!"
        return self.mask
    

    @classmethod
    def load(cls, path, load_list):
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

        # create Datastore instance
        return cls(path, datas)


    def dump(self, verbose=True, dump_list=None):
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
        write_config(self.path, config)


    def load_faiss_index(self, filename, move_to_gpu=True, verbose=True):
        r"""
        load faiss index from disk

        Args:
            filename: the prefix of faiss_index file, for example `keys.faiss_index`, filename is `keys`
            move_to_gpu: wether move the faiss index to GPU
        """
        index_path = os.path.join(self.path, filename+".faiss_index")
        # we open config file and get the shape
        config = read_config(self.path)
        
        if not hasattr(self, "faiss_index") or self.faiss_index is None:
            self.faiss_index = {}
        self.faiss_index[filename] = load_faiss_index(
                        path = index_path,
                        n_probe = 32,
                        move_to_gpu = move_to_gpu,
                        verbose=verbose
                        )


    def build_faiss_index(self, name, verbose=True, do_pca=False, pca_dim=256, use_gpu=True):
        r"""
        build faiss index for a data.
        the output file named name+.faiss_index

        Args:
            name: The data name which need to build faiss index
            verbose: display detailed message
            do_pca: wether do a PCA when building faiss index
            pca_dim: if use PCA, the PCA output dim
        """

        if not isinstance(self.datas[name], Memmap):
            print("ERROR: can only build faiss for Memmap object.")
            os.exit(1)
        # build faiss
        build_faiss_index(
                    self.datas[name].data, 
                    self.datas[name].shape,
                    os.path.join(self.path, name+".faiss_index"),
                    do_pca=do_pca,
                    pca_dim=pca_dim,
                    use_gpu=use_gpu,
                    verbose=verbose
                    )


 
