import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepCAD.model.model_utils import _get_padding_mask, _get_visibility_mask
from DeepCAD.cadlib.macro import *


class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long())
        loss_args = F.cross_entropy(args_logits[mask.bool()].reshape(-1, self.args_dim), tgt_args[mask.bool()].reshape(-1).long() + 1)  # shift due to -1 PAD_VAL

        # the loss above is for all the continious values that has been descritized, the loss below is for the real descrite values
        ext_indices = torch.where(tgt_commands == EXT_IDX)
        arc_indices = torch.where(tgt_commands == ARC_IDX)

        ext_logits = args_logits[ext_indices]
        arc_logits = args_logits[arc_indices]
        tgt_ext = tgt_args[ext_indices]
        tgt_arc = tgt_args[arc_indices]
    
        boolean_logits = ext_logits[:, EXTRUDE_OPERATION_IDX, 1:NUM_EXTRUE_OPERATIONS+1] # shift due to -1 PAD_VAL
        type_logits = ext_logits[:, EXTENT_TYPE_IDX, 1:NUM_EXTENT_TYPE+1]
        flag_logits = arc_logits[:, FLAG_IDX, 1:2+1]
        tgt_boolean = tgt_ext[:, EXTRUDE_OPERATION_IDX]
        tgt_type = tgt_ext[:, EXTENT_TYPE_IDX]
        tgt_flag = tgt_arc[:, FLAG_IDX]
    
        loss_boolean = F.cross_entropy(boolean_logits, tgt_boolean.long(), reduction='sum')  # not doing +1 due to handling this above 
        loss_type = F.cross_entropy(type_logits, tgt_type.long(), reduction='sum')
        loss_flag = F.cross_entropy(flag_logits, tgt_flag.long(), reduction='sum')
        total_discrete = len(tgt_boolean) + len(tgt_type) + len(tgt_flag)
        loss_discrete = (loss_boolean + loss_type + loss_flag) / total_discrete

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args
        loss_discrete = self.weights["loss_args_weight"] * loss_discrete

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res
