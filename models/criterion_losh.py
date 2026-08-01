import torch
import torch.nn.functional as F
from torch import nn


from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)
from einops import rearrange

class SetCriterion(nn.Module):
    """ This class computes the loss for SgMg.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, args, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 1
        self.mask_out_stride_low = self.mask_out_stride * 2

    # t*q labels 0/1 indicates whether is a blank frame.
    def loss_labels(self, long_outputs, short_outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in long_outputs
        assert 'pred_logits' in short_outputs
        long_src_logits = long_outputs['pred_logits']
        short_src_logits = short_outputs['pred_logits']
        _, nf, nq = long_src_logits.shape[:3]
        long_src_logits = rearrange(long_src_logits, 'b t q k -> b (t q) k')
        short_src_logits = rearrange(short_src_logits, 'b t q k -> b (t q) k')
        # judge the valid frames
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices):
            valid_ind = valid.nonzero().flatten()
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(long_src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=long_src_logits.device)
        if self.num_classes == 1:
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([long_src_logits.shape[0], long_src_logits.shape[1], long_src_logits.shape[2] + 1],
                                            dtype=long_src_logits.dtype, layout=long_src_logits.layout, device=long_src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        long_loss_ce = sigmoid_focal_loss(long_src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                          gamma=2) * long_src_logits.shape[1]
        short_loss_ce = sigmoid_focal_loss(short_src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                          gamma=2) * short_src_logits.shape[1]
        losses = {'loss_ce': (long_loss_ce + short_loss_ce)}

        if log:
            pass
        return losses

    def loss_boxes(self, long_outputs, short_outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert 'pred_boxes' in long_outputs
        assert 'pred_boxes' in short_outputs
        long_src_boxes = long_outputs['pred_boxes']
        short_src_boxes = short_outputs['pred_boxes']
        #print(long_src_boxes.shape, short_src_boxes.shape) #1 5 5 4
        bs, nf, nq = long_src_boxes.shape[:3]
        long_src_boxes = long_src_boxes.transpose(1, 2)
        short_src_boxes = short_src_boxes.transpose(1, 2)

        idx = self._get_src_permutation_idx(indices)
        #print(idx) #tensor[0], tensor[1]
        long_src_boxes = long_src_boxes[idx]
        # print(long_src_boxes.shape, "long_src_boxes_idx") #1 5 4
        long_src_boxes = long_src_boxes.flatten(0, 1)
        short_src_boxes = short_src_boxes[idx]
        # print(short_src_boxes.shape, "short_src_boxes_idx")#1 5 4
        short_src_boxes = short_src_boxes.flatten(0, 1)

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)
        # print(target_boxes.shape, 'target_boxes') #5 4
        long_loss_bbox = F.l1_loss(long_src_boxes, target_boxes, reduction='none')
        short_loss_bbox = F.l1_loss(short_src_boxes, target_boxes, reduction='none')
        # print(long_loss_bbox) #1 5 4
        # print(short_loss_bbox) #1 5 4

        losses = {}
        losses['loss_bbox'] = (long_loss_bbox.sum() + short_loss_bbox.sum()) / (num_boxes)
        # print(losses['loss_bbox']) #1 1462

        long_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(long_src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        short_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(short_src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = (long_loss_giou.sum() + short_loss_giou.sum()) / (num_boxes)
        return losses

    def loss_masks(self, long_outputs, short_outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in long_outputs
        assert "pred_masks" in short_outputs

        long_src_masks = long_outputs["pred_masks"]
        short_src_masks = short_outputs["pred_masks"]
        long_src_masks_low = long_outputs["pred_masks_low"]
        short_src_masks_low = short_outputs["pred_masks_low"]

        # future use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets],
                                                              size_divisibility=32, split=False).decompose()
        long_target_masks = target_masks.to(long_src_masks_low)
        short_target_masks = target_masks.to(short_src_masks_low)

        start = int(self.mask_out_stride // 2)
        start_low = int(self.mask_out_stride_low // 2)
        im_h, im_w = long_target_masks.shape[-2:]
        long_target_masks_low = long_target_masks[:, :, start_low::self.mask_out_stride_low, start_low::self.mask_out_stride_low]
        short_target_masks_low = short_target_masks[:, :, start_low::self.mask_out_stride_low, start_low::self.mask_out_stride_low]
        long_target_masks = long_target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        short_target_masks = short_target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]

        assert long_target_masks.size(2) * self.mask_out_stride == im_h
        assert long_target_masks.size(3) * self.mask_out_stride == im_w
        assert short_target_masks.size(2) * self.mask_out_stride == im_h
        assert short_target_masks.size(3) * self.mask_out_stride == im_w

        long_src_masks = long_src_masks.flatten(1)
        long_target_masks = long_target_masks.flatten(1)
        short_src_masks = short_src_masks.flatten(1)
        short_target_masks = short_target_masks.flatten(1)

        long_src_masks_low = long_src_masks_low.flatten(1)
        long_target_masks_low = long_target_masks_low.flatten(1)
        short_src_masks_low = short_src_masks_low.flatten(1)
        short_target_masks_low = short_target_masks_low.flatten(1)

        losses = {
            "loss_mask": (sigmoid_focal_loss(long_src_masks, long_target_masks, num_boxes)+
                         sigmoid_focal_loss(short_src_masks, short_target_masks, num_boxes)),
            "loss_dice": (dice_loss(long_src_masks, long_target_masks, num_boxes)+
                         dice_loss(short_src_masks, short_target_masks, num_boxes)),
            "loss_mask_low": (sigmoid_focal_loss(long_src_masks_low, long_target_masks_low, num_boxes)+
                             sigmoid_focal_loss(short_src_masks_low, short_target_masks_low, num_boxes)),
            "loss_dice_low": (dice_loss(long_src_masks_low, long_target_masks_low, num_boxes)+
                             dice_loss(short_src_masks_low, short_target_masks_low, num_boxes))
        }
        return losses

    #todo
    def loss_conditioned_iou(self, long_outputs, short_outputs, targets, indices, num_boxes):
        assert "pred_masks" in long_outputs
        assert "pred_masks" in short_outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        long_src_masks = long_outputs["pred_masks"]
        long_src_masks = long_src_masks[src_idx]
        short_src_masks = short_outputs["pred_masks"]
        short_src_masks = short_src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(long_src_masks)
        target_masks = target_masks[tgt_idx]

        # print("before____long_src_masks: ", long_src_masks.shape) #[Num_instances, H/4, W/4]
        # print("before____short_src_masks: ", short_src_masks.shape)

        long_src_masks = interpolate(long_src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear",
                                     align_corners=False)
        long_src_masks = long_src_masks[:, 0].flatten(1).sigmoid()
        short_src_masks = interpolate(short_src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear",
                                      align_corners=False)
        short_src_masks = short_src_masks[:, 0].flatten(1).sigmoid()

        # print("after____long_src_masks: ", long_src_masks.shape) #[Num_instances, H*W]
        # print("after____short_src_masks: ", short_src_masks.shape)

        long_pred_masks = long_src_masks > 0.5
        short_pred_masks = short_src_masks > 0.5
        long_src_masks = long_src_masks * long_pred_masks
        short_src_masks = short_src_masks * short_pred_masks

        numerator = (long_src_masks * short_src_masks).sum(-1)
        denominator = long_src_masks.sum(-1)
        loss = ((numerator + 1.0) / (denominator + 1.0)).mean(0)
        loss = 1 - loss
        losses = {'loss_conditioned_iou': loss}

        # print('c: ', losses['loss_conditioned_iou'].shape)
        # print(losses['loss_conditioned_iou'])

        return losses



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, long_outputs, short_outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'conditioned_iou': self.loss_conditioned_iou,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](long_outputs, short_outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        #todo
        long_outputs, short_outputs = outputs

        # use hungarian matching results.
        indices = long_outputs["main_matcher_index"]
        aux_indices = long_outputs["aux_matcher_index"]

        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T] tensor([1, 1, 0, 0, 0]
        num_boxes = target_valid.sum().item()
        device = long_outputs['pred_masks_low'].device
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():  # True
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()  # 5

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, long_outputs, short_outputs, targets, indices, num_boxes))

        #print(aux_indices)
        #[[(tensor([4], device='cuda:0'), tensor([0], device='cuda:0'))], [(tensor([4], device='cuda:0'), tensor([0], device='cuda:0'))], [(tensor([1], device='cuda:0'), tensor([0], device='cuda:0'))]]
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer. change outputs->aux_outputs
        if 'aux_outputs' in long_outputs:
            assert len(aux_indices) == len(long_outputs['aux_outputs']), "Aux index len not match." #3
            for i, (long_aux_outputs, short_aux_outputs) in enumerate(zip(long_outputs['aux_outputs'], short_outputs['aux_outputs'])):
                indices = aux_indices[i]
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, long_aux_outputs, short_aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


