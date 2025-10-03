import pdb
import shutil
import sys
import torch
from veridiq.linear_probing.train_test import get_checkpoint_path_from_folder

feature_type = sys.argv[1]
folder = f"/data/av-datasets/ckpts_linear_probing/ckpts/{feature_type}"
path_orig = get_checkpoint_path_from_folder(folder)
path_back = path_orig + ".bak"

ckpt = torch.load(path_orig)

if "feats_dim" in ckpt["hyper_parameters"]["config"]["model_hparams"]:
    print(feature_type, "already has feats_dim")
else:
    feats_dim = ckpt["state_dict"]["head.weight"].shape[1]
    ckpt["hyper_parameters"]["config"]["model_hparams"]["feats_dim"] = feats_dim

    print(feature_type, feats_dim)
    shutil.copy(path_orig, path_back)
    torch.save(ckpt, path_orig)