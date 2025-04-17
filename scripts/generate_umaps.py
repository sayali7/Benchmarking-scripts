import tqdm
import pandas as pd
import scanpy as sc
import anndata as ad

#### for nvidia rapids-singlecell
'''
import rapids_singlecell as rsc
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(
    managed_memory=False,  # Allows oversubscription
    pool_allocator=False,  # default is False
    devices=0,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)
'''
####

meta=pd.read_csv("/media/sayalialatkar/T9/Sayali/FoundationModels/PsychAD_2_5/MSSM/MSSM_meta_obs.csv")
meta.head()

donors = meta["SubID"].unique()
use_rep_x = True

meth1_save_path = "/media/sayalialatkar/T9/Sayali/FoundationModels/scGPT-main/results/zero-shot/extracted_cell_embeddings_full_body/"
meth2_save_path ="/media/sayalialatkar/T9/Sayali/FoundationModels/scMulan-main/results/zero-shot/extracted_cell_embeddings/"
meth3_save_path = "/media/sayalialatkar/T9/Sayali/FoundationModels/UCE_venv/UCE-main/results/cell_embeddings/extracted_cell_embeddings/"
meth4_save_path  = "/media/sayalialatkar/T9/Sayali/FoundationModels/scFoundation/extracted_cell_embeddings/"
meth5_save_path  = "/media/sayalialatkar/T9/Sayali/FoundationModels/Geneformer_30M/extracted_cell_embeddings/"

methods=["scGPT","scMulan","UCE","scFoundation","Geneformer"]
all_paths = [meth1_save_path, meth2_save_path, meth3_save_path, meth4_save_path, meth5_save_path]

import matplotlib.pyplot as plt

for i in range(1,len(methods)):
    #print (methods[i])
    cells = pd.DataFrame()
    for d in tqdm.tqdm(donors):
        df = pd.read_csv(all_paths[i]+d+".csv", index_col=0)
        cells = pd.concat([cells,df])
    cells.index = meta.index
    adata = ad.AnnData(cells,obs=meta,var=cells.columns.to_frame())

    ####  
    # rsc.get.anndata_to_GPU(adata)
    # if use_rep_x:
        # rsc.pp.neighbors(adata,use_rep='X')
    # else:
        # rsc.pp.pca(adata, n_comps=100, use_highly_variable=False)
        # rsc.pp.neighbors(adata)
    # rsc.tl.umap(adata, min_dist=0.3)
    ####

    if use_rep_x:
        sc.pp.neighbors(adata,use_rep='X',)
    else:
        sc.pp.pca(adata, n_comps=100, use_highly_variable=False)
        sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(adata, 
            color=['class', 'subclass',], 
            frameon=False, 
            wspace=0.4, 
            size=0.8,
            title=[f"{methods[i]}: class", f"{methods[i]}: subclass"])
    plt.savefig(f"./{methods[i]}.png", dpi=300, bbox_inches='tight')

    del cells 
    del adata