from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
from cadlib.macro import *
from utils import read_ply


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    if config.train_pc2cad:
        dataset = PCCADDataset(phase, config)
    else:
        dataset = CADDataset(phase, config)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


class CADDataset(Dataset):
    def __init__(self, phase, config):
        super(CADDataset, self).__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.aug = config.augment
        self.path = os.path.join(config.data_root, "train_val_test_split.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        if self.aug and self.phase == "train":
            command1 = cad_vec[:, 0]
            ext_indices1 = np.where(command1 == EXT_IDX)[0]
            # if len(ext_indices1) > 1 and random.randint(0, 1) == 1:
            if len(ext_indices1) > 1 and random.uniform(0, 1) > 0.5:
                ext_vec1 = np.split(cad_vec, ext_indices1 + 1, axis=0)[:-1]
        
                data_id2 = self.all_data[random.randint(0, len(self.all_data) - 1)]
                h5_path2 = os.path.join(self.raw_data, data_id2 + ".h5")
                with h5py.File(h5_path2, "r") as fp:
                    cad_vec2 = fp["vec"][:]
                command2 = cad_vec2[:, 0]
                ext_indices2 = np.where(command2 == EXT_IDX)[0]
                ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]
        
                n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
                old_idx = sorted(random.sample(list(range(len(ext_vec1))), n_replace))
                new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))
                for i in range(len(old_idx)):
                    ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]
        
                sum_len = 0
                new_vec = []
                for i in range(len(ext_vec1)):
                    sum_len += len(ext_vec1[i])
                    if sum_len > self.max_total_len:
                        break
                    new_vec.append(ext_vec1[i])
                cad_vec = np.concatenate(new_vec, axis=0)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "id": data_id}

    def __len__(self):
        return len(self.all_data)


class PCCADDataset(CADDataset):
    def __init__(self, phase, config):
        super(PCCADDataset, self).__init__(phase, config)
        self.n_points = config.n_points
        self.pc_root = config.pc_root
        
        real_data_idx = []
        real_data = []
        for idx, data_id in enumerate(self.all_data):
            pc_path = os.path.join(self.pc_root, data_id + '.ply')
            if not os.path.exists(pc_path):
                continue
            real_data_idx.append(idx)    
            real_data.append(data_id)    

        if len(real_data) == 0 and phase == 'validation':
            with open(self.path, "r") as fp:
                self.all_data = json.load(fp)['test']

            for idx, data_id in enumerate(self.all_data[: len(self.all_data)//2]):
                pc_path = os.path.join(self.pc_root, data_id + '.ply')
                if not os.path.exists(pc_path):
                    continue
                real_data_idx.append(idx)    
                real_data.append(data_id)    

        self.all_data = real_data
        assert len(self.all_data) > 0, 'Dataset is empty.'

    def __getitem__(self, index):
        all_but_points = super().__getitem__(index)

        data_id = all_but_points['id']
        pc_path = os.path.join(self.pc_root, data_id + '.ply')
        if not os.path.exists(pc_path):
            raise ValueError(f'{pc_path} not found!')
        pc = read_ply(pc_path)
        sample_idx = random.sample(list(range(pc.shape[0])), self.n_points)
        pc = pc[sample_idx]
        points = torch.tensor(pc, dtype=torch.float32)

        all_but_points['points'] = points
        return all_but_points
