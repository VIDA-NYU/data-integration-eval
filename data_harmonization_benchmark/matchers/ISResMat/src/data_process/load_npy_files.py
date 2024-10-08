import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_samples
from tqdm import tqdm

npy_dir = "../../data/output/xxxxxx/np_records"

center_npy_cnt = 0
agent_npy_cnt = 0

all_src_agent_sh_ls = []
all_tgt_agent_sh_ls = []
all_src_center_sh_ls = []
all_tgt_center_sh_ls = []
src_dataname_ls = []
tgt_dataname_ls = []


def deal_data(
    src_final_repr,
    src_flat_label_ls,
    tgt_final_repr,
    tgt_flat_label_ls,
    src_agents,
    tgt_agents,
    data_dir,
):
    src_n_cols = len(src_flat_label_ls.unique())
    src_centers = torch.empty(
        (src_n_cols, src_final_repr.shape[1]), dtype=src_final_repr.dtype
    )
    for label in range(src_n_cols):
        label_mask = src_flat_label_ls == label
        label_embeddings = src_final_repr[label_mask]
        average_embedding = torch.mean(label_embeddings, dim=0)
        average_embedding = F.normalize(average_embedding, dim=0)
        src_centers[label] = average_embedding

    tgt_n_cols = len(tgt_flat_label_ls.unique())
    tgt_centers = torch.empty(
        (tgt_n_cols, tgt_final_repr.shape[1]), dtype=tgt_final_repr.dtype
    )
    for label in range(tgt_n_cols):
        label_mask = tgt_flat_label_ls == label
        label_embeddings = tgt_final_repr[label_mask]
        average_embedding = torch.mean(label_embeddings, dim=0)
        average_embedding = F.normalize(average_embedding, dim=0)
        tgt_centers[label] = average_embedding

    src_agent_sh_ls = silhouette_samples(
        torch.cat((src_final_repr, src_agents)),
        torch.cat((src_flat_label_ls, torch.tensor(list(range(src_agents.shape[0]))))),
        metric="cosine",
    )[-src_agents.shape[0] :]
    src_center_sh_ls = silhouette_samples(
        torch.cat((src_final_repr, src_centers)),
        torch.cat((src_flat_label_ls, torch.tensor(list(range(src_agents.shape[0]))))),
        metric="cosine",
    )[-src_agents.shape[0] :]
    tgt_agent_sh_ls = silhouette_samples(
        torch.cat((tgt_final_repr, tgt_agents)),
        torch.cat((tgt_flat_label_ls, torch.tensor(list(range(tgt_agents.shape[0]))))),
        metric="cosine",
    )[-tgt_agents.shape[0] :]
    tgt_center_sh_ls = silhouette_samples(
        torch.cat((tgt_final_repr, tgt_centers)),
        torch.cat((tgt_flat_label_ls, torch.tensor(list(range(tgt_agents.shape[0]))))),
        metric="cosine",
    )[-tgt_agents.shape[0] :]
    global all_src_agent_sh_ls
    global all_tgt_agent_sh_ls
    global all_src_center_sh_ls
    global all_tgt_center_sh_ls
    global src_dataname_ls
    global tgt_dataname_ls

    all_src_agent_sh_ls += src_agent_sh_ls.tolist()
    all_tgt_agent_sh_ls += tgt_agent_sh_ls.tolist()
    all_src_center_sh_ls += src_center_sh_ls.tolist()
    all_tgt_center_sh_ls += tgt_center_sh_ls.tolist()

    data_dir = data_dir.replace("-", "/")
    src_dataname_ls += [data_dir] * len(src_agent_sh_ls.tolist())
    tgt_dataname_ls += [data_dir] * len(tgt_agent_sh_ls.tolist())
    # print('src agent sh mean:', src_agent_sh_mean)
    # print('tgt agent sh mean:', tgt_agent_sh_mean)
    # print('src center sh mean:', src_center_sh_mean)
    # print('tgt center sh mean:', tgt_center_sh_mean)


btime = time.time()
for dir in tqdm(os.listdir(npy_dir)):
    data_pair_dir = os.path.join(npy_dir, dir)
    for files in os.listdir(data_pair_dir):
        file_path = os.path.join(data_pair_dir, files)
        if "src_val_points.npy" in file_path:
            src_final_repr = torch.from_numpy(np.load(file_path))
        elif "src_val_points_labels.npy" in file_path:
            src_flat_label_ls = torch.from_numpy(np.load(file_path))
        elif "tgt_val_points.npy" in file_path:
            tgt_final_repr = torch.from_numpy(np.load(file_path))
        elif "tgt_val_points_labels.npy" in file_path:
            tgt_flat_label_ls = torch.from_numpy(np.load(file_path))
        elif "src_agents.npy" in file_path:
            src_agents = torch.from_numpy(np.load(file_path))
        elif "tgt_agents.npy" in file_path:
            tgt_agents = torch.from_numpy(np.load(file_path))

    deal_data(
        src_final_repr,
        src_flat_label_ls,
        tgt_final_repr,
        tgt_flat_label_ls,
        src_agents,
        tgt_agents,
        dir,
    )


etime = time.time()
print("time:", etime - btime)

all_agents_sh_ls = all_src_agent_sh_ls + all_tgt_agent_sh_ls
all_center_sh_ls = all_src_center_sh_ls + all_tgt_center_sh_ls
all_dataname_ls = src_dataname_ls + tgt_dataname_ls

assert len(all_dataname_ls) == len(all_agents_sh_ls)
assert len(all_dataname_ls) == len(all_center_sh_ls)

all_agents_sh_mean = sum(all_agents_sh_ls) / len(all_agents_sh_ls)
all_center_sh_mean = sum(all_center_sh_ls) / len(all_center_sh_ls)


print(all_agents_sh_mean)
print(all_center_sh_mean)
