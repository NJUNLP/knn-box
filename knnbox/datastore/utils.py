r""" some utils function for building datastore"""
import os
import faiss
import numpy as np
import time
import ctypes

def build_faiss_index(
                keys,
                shape,
                output_filename,
                train_index_count = 1000000,
                n_centroids = 4096,
                code_size = 64,
                n_probe = 32,
                num_keys_to_add_at_a_time = 500000,
                seed = 1,
                do_pca = False,
                pca_dim = 256, # if do_pca==True, reduce to pca_dim before faiss retrieve
                use_gpu = True, # use faiss-gpu, othewise use faiss-cpu
                verbose=False
                ):
    r""" 
    build faiss index for a memmap
    this function is mostly inspired from kNN-LM code 
    """
    print("[Start Building Faiss Index]")
    # set OMP_WAIT_POLICY=PASSIVE significantly speed up faiss-cpu
    os.environ["OMP_WAIT_POLICY"]="PASSIVE"

    res = faiss.StandardGpuResources()
    capacity, dimension = shape
    progress_idx = 1
    total_progress = 3
    if do_pca:
        total_progress += 1
    if use_gpu:
        total_progress += 1


    if not os.path.exists(output_filename+".trained"):
        index_dim = pca_dim if do_pca else dimension
        quantizer = faiss.IndexFlatL2(index_dim)
        index = faiss.IndexIVFPQ(quantizer, index_dim, n_centroids, code_size, 8)
        index.nprobe = n_probe


        if use_gpu:
            start = time.time() 
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
            if verbose:
                print("  > [{}/{}] put index to GPU took {} s". \
                    format(progress_idx, total_progress, time.time()-start))
                progress_idx += 1

        # if use PCA, wrap the index with pre-PCA operation
        if do_pca == True:
            print("  > [{}/{}] do pca operation".format(progress_idx, total_progress))
            start = time.time()
            pca_matrix = faiss.PCAMatrix(dimension, pca_dim, 0, True)
            if not use_gpu:
                index = faiss.IndexPreTransform(pca_matrix, index)
            else:
                gpu_index = faiss.IndexPreTransform(pca_matrix, gpu_index)
            if verbose:
                print("  > [{}/{}] pca operation took {} s".\
                    format(progress_idx, total_progress, time.time()-start))
                progress_idx += 1
        
        if verbose:
            print("  > [{}/{}] training index (about 3 minutes)...".format(progress_idx, total_progress))
        start = time.time()
        np.random.seed(seed)
        random_sample = np.random.choice(
            np.arange(capacity), size = [min(train_index_count, capacity)],
            replace=False
        )

        # faiss dosent handle train keys in fp16, so convert to fp32 first
        if use_gpu:
            gpu_index.train(keys[random_sample].astype(np.float32))
        else:
            index.train(keys[random_sample].astype(np.float32))
        
        if verbose:
            print("  > [{}/{}] training took {} s".format(progress_idx, total_progress, time.time() - start))
            progress_idx += 1
            print("  > [{}/{}] writing index after training...".format(progress_idx, total_progress))
        start = time.time()
        if use_gpu:
            faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename+".trained")
        else:
            faiss.write(index, output_filename+".trained")
        if verbose:
            print("  > [{}/{}] writing index took {} s".format(progress_idx, total_progress, time.time() -start))
            progress_idx += 1
    
    if verbose:
        print("  > [{}/{}] adding keys...".format(progress_idx, total_progress))
    # read the trained model
    index = faiss.read_index(output_filename + ".trained")
    if use_gpu:
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
    start = 0
    start_time = time.time()

    while start < capacity:
        end = min(capacity, start+num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()
        if use_gpu:
            gpu_index.add_with_ids(to_add.astype(np.float32), np.arange(start,end))
        else:
            index.add_with_ids(to_add.astype(np.float32), np.arange(start,end))
        start += num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            if verbose:
                print("  > [{}/{}] added {} tokens so far, total {}.".format(
                            progress_idx, total_progress,min(start, capacity), capacity))
            # faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename)
            
    
    if verbose:
        start = time.time()
        if verbose:
            print("  > [{}/{}] adding total {} keys ".format(progress_idx, total_progress, end))
            print("  > [{}/{}] adding took {} s".format(progress_idx, total_progress, time.time() - start_time))
    if use_gpu:
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename)
    else:
        faiss.write_index(index, output_filename)

    index.reset()
    del index
    if use_gpu:
        gpu_index.reset()
        del gpu_index
    # remove the temporary trained index
    if os.path.exists(output_filename+".trained"):
        os.remove(output_filename+".trained")
    print("[Finish Building Faiss Index  Successfully ^_^]")



def load_faiss_index(path, n_probe,
            move_to_gpu=True, verbose=False):
    r"""
    load the faiss index"""
    print("[Start Loading Faiss Index]")
    if verbose:
        start_time = time.time()
    
    # check if the faiss index has been built
    if not os.path.exists(path):
        print("!!Error: faiss index hasn't beed built, Pleast built it first and then load it")
        import sys
        sys.exit(1)
 
    index = faiss.read_index(path, faiss.IO_FLAG_ONDISK_SAME_DIR)
    if move_to_gpu:
        if verbose:
            print("  > move faiss index to gpu")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
    if verbose:
        print("  > reading index took {} s".format(time.time()-start_time))
        # Print all vector transforms
        if isinstance(index,faiss.swigfaiss_avx2.IndexPreTransform):
            print(f"    > {index.chain.size()} pre-transform is found")
            for i in range(index.chain.size()):
                print(f"    > pre-transform: {index.chain.at(i).d_in} -> {index.chain.at(i).d_out}")
        print("  > the datastore shape is ", (index.ntotal, index.d))
    index.nprobe = n_probe
    print("[Finish Loading Faiss Index Successfully ^_^]")
    return index
