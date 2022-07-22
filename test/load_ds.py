
from libds.datastore.datastore import Datastore
from libds.retriever.retriever import Retriever
from libds.integrator.integrator import Integrator
import torch

ds = Datastore.load("/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_datastore")
retriever = Retriever(ds, 8) 
integrator = Integrator(0.7, 10, 32000)

if __name__ == "__main__":
    query = torch.randn(32,1024)
    result = retriever.retrieve(query)
    print(result)

    knn_probs = integrator.get_knn_prob(**result, device=query.device)
    integ_probs = integrator.get_integrated_prob(knn_probs, knn_probs)
    print (integ_probs)
