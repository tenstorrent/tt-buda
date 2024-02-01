# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import math
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bbox_iou, multi_bbox_ious


class YoloLayer(nn.Module):
    def __init__(self, anchors, stride, num_classes):
        super().__init__()
        self.anchors, self.stride = np.array(anchors), stride
        self.num_classes = num_classes

    def get_masked_anchors(self):
        return self.anchors / self.stride

    def get_region_boxes(self, output, conf_thresh):
        if output.dim() == 3:
            output = output.unsqueeze(0)
        device = output.device  # torch.device(torch_device)
        anchors = torch.from_numpy(self.get_masked_anchors().astype(np.float32)).to(
            device
        )

        nB = output.size(0)
        nA = len(anchors)
        nC = self.num_classes
        nH = output.size(2)
        nW = output.size(3)
        cls_anchor_dim = nB * nA * nH * nW

        assert output.size(1) == (5 + nC) * nA

        # if you want to debug this is how you get the indexes where objectness is high
        # output = output.view(nB, nA, 5+nC, nH, nW)
        # inds = torch.nonzero((torch.sigmoid(output.view(nB, nA, 5+nC, nH, nW)[:,:,4,:,:]) > conf_thresh))

        output = (
            output.view(nB * nA, 5 + nC, nH * nW)
            .transpose(0, 1)
            .contiguous()
            .view(5 + nC, cls_anchor_dim)
        )

        grid_x = (
            torch.linspace(0, nW - 1, nW)
            .repeat(nB * nA, nH, 1)
            .view(cls_anchor_dim)
            .to(device)
        )
        grid_y = (
            torch.linspace(0, nH - 1, nH)
            .repeat(nW, 1)
            .t()
            .repeat(nB * nA, 1, 1)
            .view(cls_anchor_dim)
            .to(device)
        )
        ix = torch.LongTensor(range(0, 2)).to(device)
        anchor_w = (
            anchors.index_select(1, ix[0]).repeat(1, nB, nH * nW).view(cls_anchor_dim)
        )
        anchor_h = (
            anchors.index_select(1, ix[1]).repeat(1, nB, nH * nW).view(cls_anchor_dim)
        )

        xs, ys = torch.sigmoid(output[0]) + grid_x, torch.sigmoid(output[1]) + grid_y
        ws, hs = (
            torch.exp(output[2]) * anchor_w.detach(),
            torch.exp(output[3]) * anchor_h.detach(),
        )
        det_confs = torch.sigmoid(output[4])

        cls_confs = torch.nn.Softmax(dim=1)(output[5 : 5 + nC].transpose(0, 1)).detach()
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        cls_max_ids = cls_max_ids.view(-1)

        det_confs = det_confs.to("cpu")  # , non_blocking=True for torch 4.1?
        cls_max_confs = cls_max_confs.to("cpu")
        cls_max_ids = cls_max_ids.to("cpu")
        xs, ys = xs.to("cpu"), ys.to("cpu")
        ws, hs = ws.to("cpu"), hs.to("cpu")

        all_boxes = [[] for i in range(nB)]

        inds = torch.LongTensor(range(0, len(det_confs)))
        for ind in inds[det_confs > conf_thresh]:
            bcx = xs[ind]
            bcy = ys[ind]
            bw = ws[ind]
            bh = hs[ind]
            # box = [bcx/nW, bcy/nH, bw/nW, bh/nH, det_confs[ind], cls_max_confs[ind], cls_max_ids[ind]]
            box = [
                bcx / nW,
                bcy / nH,
                bw / nW,
                bh / nH,
                det_confs[ind],
                cls_max_confs[ind],
                cls_max_ids[ind],
            ]
            box = [i.item() for i in box]

            batch = math.ceil(ind / (nA * nH * nW))
            all_boxes[batch].append(box)

        return all_boxes

    def build_targets(self, pred_boxes, target, anchors, nH, nW):
        self.ignore_thresh = 0.5
        self.truth_thresh = 1.0

        # Works faster on CPU than on GPU.
        devi = torch.device("cpu")
        pred_boxes = pred_boxes.to(devi)
        target = target.to(devi)
        anchors = anchors.to(devi)

        # max_targets = target[0].view(-1,5).size(0) # 50
        nB = target.size(0)
        nA = len(anchors)

        anchor_step = anchors.size(1)  # anchors[nA][anchor_step]
        conf_mask = torch.ones(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)
        # twidth, theight = self.net_width/self.stride, self.net_height/self.stride
        twidth, theight = nW, nH
        nAnchors = nA * nH * nW

        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors : (b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1, 5)

            # If the bounding box prior is not the best but does overlap a ground truth object by
            # more than some threshold we ignore the prediction (conf_mask)
            for t in range(tbox.size(0)):
                if tbox[t][1] == 0:
                    break
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * twidth, tbox[t][4] * theight
                cur_gt_boxes = (
                    torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                )
                cur_ious = torch.max(
                    cur_ious,
                    multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False),
                )
            ignore_ix = cur_ious > self.ignore_thresh
            conf_mask[b][ignore_ix.view(nA, nH, nW)] = 0

            for t in range(tbox.size(0)):
                if tbox[t][1] == 0:
                    break
                # nGT += 1
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * twidth, tbox[t][4] * theight
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)

                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, anchor_step), anchors), 1).t()
                _, best_n = torch.max(
                    multi_bbox_ious(tmp_gt_boxes, anchor_boxes, x1y1x2y2=False), 0
                )

                coord_mask[b][best_n][gj][gi] = 1
                cls_mask[b][best_n][gj][gi] = 1
                conf_mask[b][best_n][gj][gi] = 1
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])
                tcls[b][best_n][gj][gi] = tbox[t][0]
                tconf[b][best_n][gj][gi] = 1  # yolov1 would have used iou-value here

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    def get_loss(self, output, target, return_single_value=True):
        device = output.device

        anchors = torch.from_numpy(self.get_masked_anchors().astype(np.float32)).to(
            device
        )

        nB = output.data.size(0)  # batch size
        nA = len(anchors)
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls_anchor_dim = nB * nA * nH * nW

        output = output.view(nB, nA, (5 + nC), nH, nW)

        ix = torch.LongTensor(range(0, 5)).to(device)
        coord = (
            output.index_select(2, ix[0:4])
            .view(nB * nA, -1, nH * nW)
            .transpose(0, 1)
            .contiguous()
            .view(4, cls_anchor_dim)
        )  # x, y, w, h
        coord[0:2] = coord[0:2].sigmoid()  # x, y:   bx = σ(tx) (+ cx)
        conf = output.index_select(2, ix[4]).view(nB, nA, nH, nW).sigmoid()

        grid_x = (
            torch.linspace(0, nW - 1, nW)
            .repeat(nB * nA, nH, 1)
            .view(cls_anchor_dim)
            .to(device)
        )
        grid_y = (
            torch.linspace(0, nH - 1, nH)
            .repeat(nW, 1)
            .t()
            .repeat(nB * nA, 1, 1)
            .view(cls_anchor_dim)
            .to(device)
        )
        anchor_w = (
            anchors.index_select(1, ix[0]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
        )
        anchor_h = (
            anchors.index_select(1, ix[1]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
        )

        pred_boxes = torch.FloatTensor(4, cls_anchor_dim).to(device)
        pred_boxes[0] = coord[0] + grid_x  # bx = σ(tx) + cx
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w  # pw*e(tw)
        pred_boxes[3] = coord[3].exp() * anchor_h
        pred_boxes = pred_boxes.transpose(0, 1).contiguous().view(-1, 4)

        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(
            pred_boxes.detach(), target.detach(), anchors.detach(), nH, nW
        )

        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long().to(device)
        cls = output.index_select(2, cls_grid)
        cls = (
            cls.view(nB * nA, nC, nH * nW)
            .transpose(1, 2)
            .contiguous()
            .view(cls_anchor_dim, nC)
        )
        cls_mask = cls_mask == 1
        tcls = tcls[cls_mask].long().view(-1)
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC).to(device)
        cls = cls[cls_mask].view(-1, nC)

        tcoord = tcoord.view(4, cls_anchor_dim).to(device)
        tconf, tcls = tconf.to(device), tcls.to(device)
        coord_mask, conf_mask = coord_mask.view(cls_anchor_dim).to(
            device
        ), conf_mask.to(device)

        loss_coord = (
            nn.MSELoss(size_average=False)(coord * coord_mask, tcoord * coord_mask) / 2
        )
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask)
        loss_cls = (
            nn.CrossEntropyLoss(size_average=False)(cls, tcls) if cls.size(0) > 0 else 0
        )
        loss = loss_coord + loss_conf + loss_cls

        if math.isnan(loss.item()):
            print(conf, tconf)
            raise ValueError("YoloLayer has isnan in loss")
            # sys.exit(0)

        if return_single_value:
            return loss
        else:
            return [loss, loss_coord, loss_conf, loss_cls]
