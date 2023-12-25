from pickle import TRUE
from matplotlib import table
from matplotlib.pyplot import box
from sympy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        # [N,2] -> [N,1,2] -> [N,M,2]
        box1[:, :2].unsqueeze(1).expand(N, M, 2),
        # [M,2] -> [1,M,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),
    )

    rb = torch.min(
        # [N,2] -> [N,1,2] -> [N,M,2]
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        # [M,2] -> [1,M,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.device = torch.device("cuda")

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then, (x,y)代表作標, (w, h)代表寬、高
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = x / self.S - .5 * w
        y1 = y / self.S - .5 * h
        x2 = x / self.S + .5 * w
        y2 = y / self.S + .5 * h
        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # size(N, 4)
        return boxes

    def find_best_iou_boxes(self, box_pred_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        4) hint: use torch.diagnoal() on results of compute_iou
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        ious = []
        best_ious = []
        best_boxes = []

        # Reshape box_pred_list tensor to size(-1, 5)
        has_object_boxes_list = box_pred_list.view(-1, 5)

        boxes_coord = self.xywh2xyxy(
            has_object_boxes_list).contiguous().view(-1, 2, 4)
        box_target = self.xywh2xyxy(box_target)

        # compute ious
        ious = [compute_iou(boxes_coord[:, i, :], box_target).diagonal()
                for i in range(self.B)]
        ious = torch.stack(ious).view(-1, self.B)  # 有 B 個 boxes
        # print("ious shape: ", ious.shape)

        # find best iou & index amoung bboxes.
        best_ious, idx = torch.max(ious, dim=1)
        # print("index: ", idx)

        best_boxes = has_object_boxes_list.view(-1, 2, 5)
        # print("best_boxes before index: ", best_boxes[:10])
        # print("best_boxes before index shape: ", best_boxes.shape)

        best_boxes = best_boxes[torch.arange(
            best_boxes.size(0)), idx, :]
        # print("best_boxes after index: ", best_boxes[:])
        # print("best_boxes after index shape: ", best_boxes.shape)

        best_boxes = best_boxes.view(-1, 5)

        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar

        https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        """
        ### CODE ###
        # Your code here
        # 只計算包含 obj 的 cell
        pred_probs = classes_pred[has_object_map].view(-1)
        target_probs = classes_target[has_object_map].view(-1)

        loss = F.mse_loss(
            pred_probs, target_probs, reduction='sum')

        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here

        no_obj_map = ~ has_object_map
        # 把每個 cell 的 bboxes 分割，以便後續的計算
        no_obj_boxes = pred_boxes_list[no_obj_map].view(-1, 2, 5)

        # 取出所有的 confidence 並 vectorize
        confidences = no_obj_boxes[:, :, -1].view(-1)
        # no object 的 grid confidence 為 0
        ground_truth = torch.zeros_like(confidences)

        loss = F.mse_loss(confidences, ground_truth, reduction="sum")
        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient
        """
        # CODE
        # your code here

        # 〔box_pred_conf〕 參數為一個 [size(-1, 1)] 的 Tensor
        # 〔box_target_conf〕參數為一個 [size(-1, 1)] 的 Tensor
        # 摺疊 tensor 的維度以符合需求(B 個 Boxes)
        pred_boxes = box_pred_conf.view(-1, 2, 5)
        confidences = pred_boxes[:, :, -1].view(-1)  # 特定出 confidence 的 column
        ground_truth = torch.ones_like(confidences)
        # 摺疊 tensor 的維度以符合需求(B 個 Boxes)
        # pred_boxes = box_pred_conf.view(-1, 2, 5)
        # confidences = pred_boxes[:, :, -1].view(-1)  # 特定出 confidence 的 column
        # ground_truth = torch.ones_like(box_pred_conf)
        loss = F.mse_loss(confidences,
                          ground_truth, reduction='sum')
        # print("contain con loss: ", loss)
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        # CODE
        # your code here
        # 計算 best box 與 ground truth 的 error, 利用 xywh 的差異來計算

        # x, y 的值都在 box_pred_resoonse, box_target_response 的前面兩個 column，直接以 indexing 的方式取出並計算MSE
        xy_loss = F.mse_loss(
            box_pred_response[:, :2], box_target_response[:, :2], reduction="sum")
        wh_loss = F.mse_loss(
            box_pred_response[:, 2:].sqrt(
            ), box_target_response[:, 2:].sqrt(), reduction="sum")
        reg_loss = xy_loss + wh_loss

        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0
        inv_N = 1.0 / N
        # When you calculate the classification loss, no-object loss, regression loss, contain_object_loss
        # you need to multiply the loss with inv_N. e.g: inv_N * self.get_regression_loss(...)

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)

        # 前面為預測的BBox, [size(24, 14, 14, 2*5)]
        pred_boxes_list = pred_tensor[..., :self.B*5]
        # 後面為預測的 cls 機率, size(24, 14, 14, 20)
        pred_cls = pred_tensor[:, :, :, self.B*5:]

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        #   1) only keep having-object cells
        #   2) vectorize all dimensions except for the last one for faster computation
        # 包含物件的bboxes: obj_boxes_list -> [size(N * 10)]
        obj_boxes_list = pred_boxes_list[has_object_map]
        target_boxes = target_boxes[has_object_map].contiguous(
        ).view(-1, 4)  # vectorize the this tensor size(-1, 4)

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        # best_boxes: 在 cell i 之中最佳的 Bounding box
        best_ious, best_boxes = self.find_best_iou_boxes(
            obj_boxes_list, target_boxes)

        # compcute classification loss, 計算包含obj的classfication loss, 第 i 個cell的class prediction probabilities
        cls_loss = self.get_class_prediction_loss(
            classes_pred=pred_cls, classes_target=target_cls, has_object_map=has_object_map) * inv_N

        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(
            pred_boxes_list=pred_boxes_list, has_object_map=has_object_map) * inv_N * self.l_noobj

        # compute contain_object_loss
        containing_obj_loss = self.get_contain_conf_loss(
            box_pred_conf=obj_boxes_list, box_target_conf=torch.ones_like(
                obj_boxes_list)
        ) * inv_N

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss = self.get_regression_loss(
            best_boxes[:, :-1], target_boxes) * inv_N * self.l_coord

        # compute final loss
        total_loss = cls_loss + no_obj_loss + reg_loss + containing_obj_loss

        # construct return loss_dict
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=containing_obj_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss,
        )
        return loss_dict
