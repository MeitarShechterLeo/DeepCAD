import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import CADTransformer, PC2CADTransformer
from .base import BaseTrainer, DecodingOnlyBaseTrainer
from .loss import CADLoss
from .scheduler import GradualWarmupScheduler
from cadlib.macro import *


class TrainerAE(BaseTrainer):
    def build_net(self, cfg):
        self.net = CADTransformer(cfg).cuda()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        if cfg.encoder_only:
            self.optimizer = optim.Adam(self.net.encoder.parameters(), cfg.lr)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

    def set_loss_function(self):
        self.loss_func = CADLoss(self.cfg).cuda()

    def forward(self, data):
        commands = data['command'].cuda() # (N, S)
        args = data['args'].cuda()  # (N, S, N_ARGS)

        outputs = self.net(commands, args)
        loss_dict = self.loss_func(outputs)

        return outputs, loss_dict

    def encode(self, data, is_batch=False):
        """encode into latent vectors"""
        commands = data['command'].cuda()
        args = data['args'].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args = args.unsqueeze(0)
        z = self.net(commands, args, encode_mode=True)
        return z

    def decode(self, z):
        """decode given latent vectors"""
        outputs = self.net(None, None, z=z, return_tgt=False)
        return outputs

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
        out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
        if refill_pad: # fill all unused element to -1
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def evaluate(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data['command'].cuda()
                args = data['args'].cuda()
                outputs = self.net(commands, args)
                out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.clock.epoch)


class DecodingOnlyTrainerAE(DecodingOnlyBaseTrainer):
    def build_net(self, cfg):
        self.net = CADTransformer(cfg)

    def decode(self, z):
        """decode given latent vectors"""
        outputs = self.net(None, None, z=z, return_tgt=False)
        return outputs

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
        out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
        if refill_pad: # fill all unused element to -1
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().to(out_args.device)[out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()

        return out_cad_vec


class TrainerPCAE(TrainerAE):
    def build_net(self, cfg):
        self.net = PC2CADTransformer(cfg).cuda()

    def load_ckpt(self, name=None, is_train=True, encoder_only_training=False):
        """load checkpoint from saved checkpoint"""
        if not encoder_only_training:
            super(TrainerPCAE, self).load_ckpt(name, is_train)

        else:
            if not os.path.exists(name):
                raise ValueError("Checkpoint {} not exists.".format(name))

            checkpoint = torch.load(name)
            print("Loading autoencoder checkpoint from {} ...".format(name))

            model_sd = checkpoint['model_state_dict']
            keys_to_load = [key for key in model_sd.keys() if "encoder" not in key]
            subset_state_dict = {key: value for key, value in model_sd.items() if key in keys_to_load}

            if isinstance(self.net, nn.DataParallel):
                self.net.module.load_state_dict(subset_state_dict, strict=False)
            else:
                self.net.load_state_dict(subset_state_dict, strict=False)

    def forward(self, data):
        commands = data['command'].cuda() # (N, S)
        args = data['args'].cuda()  # (N, S, N_ARGS)
        points = data['points'].cuda()

        outputs = self.net(points, commands, args)
        loss_dict = self.loss_func(outputs)

        return outputs, loss_dict

    def encode(self, data, is_batch=False):
        """encode into latent vectors"""
        commands = data['command'].cuda()
        args = data['args'].cuda()
        points = data['points'].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args = args.unsqueeze(0)
            points = points.unsqueeze(0)
        z = self.net(points, commands, args, encode_mode=True)
        return z

    def evaluate(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data['command'].cuda()
                args = data['args'].cuda()
                points = data['points'].cuda()
                outputs = self.net(points, commands, args)
                out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int32)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.clock.epoch)
