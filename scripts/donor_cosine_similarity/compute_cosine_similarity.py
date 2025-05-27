import os
import tqdm
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH=""
SAVE_PATH=""


meth1_save_path = f"{DATA_PATH}/scGPT/extracted_cell_embeddings_full_body/"
meth2_save_path =f"{DATA_PATH}/scMulan/extracted_cell_embeddings/"
meth3_save_path = f"{DATA_PATH}/UCE/extracted_cell_embeddings/"
meth4_save_path  = f"{DATA_PATH}/scFoundation/extracted_cell_embeddings/"
meth5_save_path  = f"{DATA_PATH}/Geneformer/extracted_cell_embeddings/"

method_save_dict = {"scGPT":meth1_save_path, "scMulan": meth2_save_path, "UCE":meth3_save_path, 
                    "scFoundation": meth4_save_path, "Geneformer": meth5_save_path}

meth1_files = os.listdir(meth1_save_path)
meth2_files = os.listdir(meth2_save_path)
meth3_files = os.listdir(meth3_save_path)
meth4_files = os.listdir(meth4_save_path)
meth5_files = os.listdir(meth5_save_path)


donor_names = []
methods = list(method_save_dict.keys())
pairwise_differences = {}

for file in tqdm.tqdm(meth1_files):
    donor = file.split(".")[0]
    df_1 = pd.read_csv(f"{meth1_save_path}{file}", index_col=0)
    df_2 = pd.read_csv(f"{meth2_save_path}{file}", index_col=0)
    df_3 = pd.read_csv(f"{meth3_save_path}{file}", index_col=0)
    df_4 = pd.read_csv(f"{meth4_save_path}{file}", index_col=0)
    df_5 = pd.read_csv(f"{meth5_save_path}{file}", index_col=0)


    emb_dict = {"scGPT":pdist(df_1.values),"scMulan":pdist(df_2.values),"UCE":pdist(df_3.values), 
                "scFoundation":pdist(df_4.values), "Geneformer": pdist(df_5.values)}
    for i,m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            #fro_norm = get_frobenius_norm(emb_dict[m1],emb_dict[m2])
            cos_sim = cosine_similarity(squareform(emb_dict[m1]),squareform(emb_dict[m2]))

            if f"{m1}_{m2}" not in pairwise_differences:
                pairwise_differences[f"{m1}_{m2}"] = [cos_sim]
            else:
                pairwise_differences[f"{m1}_{m2}"].append(cos_sim)

    donor_names.append(donor)

pairwise_differences_df = pd.DataFrame(pairwise_differences)
pairwise_differences_df["donor"] = donor_names

pairwise_differences_df.to_csv(f"{SAVE_PATH}/pairwise_donor_cosine_similarity.csv")