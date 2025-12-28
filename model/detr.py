import torch
import torch.nn as nn
import torchvision.models
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from torchvision.models import (
    resnet34,
    resnet50,
    resnet101,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)

def build_backbone(config):
    match config["backbone"]:
        case "resnet34":
            return resnet34(
                weights=ResNet34_Weights.IMAGENET1K_V1,
                norm_layer=torchvision.ops.FrozenBatchNorm2d
            )

        case "resnet50":
            return resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1,
                norm_layer=torchvision.ops.FrozenBatchNorm2d
            )

        case "resnet101":
            return resnet101(
                weights=ResNet101_Weights.IMAGENET1K_V1,
                norm_layer=torchvision.ops.FrozenBatchNorm2d
            )

        case _:
            raise ValueError(f"Backbone không hỗ trợ: {config['backbone']}")


def get_spatial_position_embedding(pos_emb_dim, feat_map):
    # Đảm bảo pos_emb_dim phải chia hết cho 4
    assert pos_emb_dim % 4 == 0, ('Position embedding dimension '
                                  'must be divisible by 4')
    grid_size_h, grid_size_w = feat_map.shape[2], feat_map.shape[3]
    grid_h = torch.arange(grid_size_h,
                          dtype=torch.float32,
                          device=feat_map.device)
    grid_w = torch.arange(grid_size_w,
                          dtype=torch.float32,
                          device=feat_map.device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_positions -> (Số lượng grid cell tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=feat_map.device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([
        torch.sin(grid_h_emb),
        torch.cos(grid_h_emb)
    ], dim=-1)
    # grid_h_emb -> (Số lượng grid cell tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([
        torch.sin(grid_w_emb),
        torch.cos(grid_w_emb)
    ], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Số lượng grid cell tokens, pos_emb_dim)
    return pos_emb


class TransformerEncoder(nn.Module):
    r"""
    Encoder cho transformer của DETR.
    Bao gồm một chuỗi các encoder layers.
    Mỗi layer có các module sau:
        1. LayerNorm cho Self Attention
        2. Self Attention
        3. LayerNorm cho MLP
        4. MLP
    """
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                  dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Module Self Attention cho tất cả các encoder layers
        self.attns = nn.ModuleList(
                [
                    nn.MultiheadAttention(d_model, num_heads,
                                          dropout=self.dropout_prob,
                                          batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        # Module MLP cho tất cả các encoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ])

        # Norm cho Self Attention cho tất cả các encoder layers
        self.attn_norms = nn.ModuleList(
                [
                    nn.LayerNorm(d_model)
                    for _ in range(num_layers)
                ])

        # Norm cho MLP cho tất cả các encoder layers
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Dropout cho Self Attention cho tất cả các encoder layers
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout cho MLP cho tất cả các encoder layers
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Norm cho encoder output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, spatial_position_embedding):
        out = x
        attn_weights = []
        for i in range(self.num_layers):
            # Norm, Self Attention, Dropout và Residual
            in_attn = self.attn_norms[i](out)
            # Thêm spatial position embedding
            # vào q, k cho self attention
            q = in_attn + spatial_position_embedding
            k = in_attn + spatial_position_embedding
            out_attn, attn_weight = self.attns[i](
                query=q,
                key=k,
                value=in_attn
            )
            attn_weights.append(attn_weight)
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, MLP, Dropout và Residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff

        # Output Normalization
        out = self.output_norm(out)
        return out, torch.stack(attn_weights)


class TransformerDecoder(nn.Module):
    r"""
        Decoder cho transformer của DETR.
        Bao gồm một chuỗi các decoder layers.
        Mỗi layer có các module sau:
            1. LayerNorm cho Self Attention
            2. Self Attention
            3. LayerNorm cho Cross Attention trên
               Encoder Outputs
            4. Cross Attention
            5. LayerNorm cho MLP
            6. MLP
    """
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                  dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Module Self Attention cho tất cả các decoder layers
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_layers)
            ])

        # Module Cross Attention cho tất cả các decoder layers
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_layers)
            ])

        # Module MLP cho tất cả các decoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ])

        # Norm cho module Self Attention cho tất cả các decoder layers
        self.attn_norms = nn.ModuleList(
                [
                    nn.LayerNorm(d_model)
                    for _ in range(num_layers)
                ])

        # Norm cho module Cross Attention cho tất cả các decoder layers
        self.cross_attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Norm cho module MLP cho tất cả các decoder layers
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Dropout cho module Attention cho tất cả các decoder layers
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout cho module Cross Attention cho tất cả các decoder layers
        self.cross_attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout cho module MLP cho tất cả các decoder layers
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # LayerNorm đầu ra dùng chung cho tất cả decoder outputs
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_objects, encoder_output,
                query_embedding, spatial_position_embedding):
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []
        for i in range(self.num_layers):
            # Norm, Self Attention, Dropout và Residual
            in_attn = self.attn_norms[i](out)
            q = in_attn + query_embedding
            k = in_attn + query_embedding
            out_attn, _ = self.attns[i](
                query=q,
                key=k,
                value=in_attn
            )
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, Cross Attention, Dropout và Residual
            in_attn = self.cross_attn_norms[i](out)
            q = in_attn + query_embedding
            k = encoder_output + spatial_position_embedding
            out_attn, decoder_cross_attn = self.cross_attns[i](
                query=q,
                key=k,
                value=encoder_output
            )
            decoder_cross_attn_weights.append(decoder_cross_attn)
            out_attn = self.cross_attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, MLP, Dropout và Residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff
            decoder_outputs.append(self.output_norm(out))

        output = torch.stack(decoder_outputs)
        return output, torch.stack(decoder_cross_attn_weights)


class DETR(nn.Module):
    r"""
    Lớp model DETR khởi tạo tất cả các layers của DETR.
    Một lượt forward pass đi qua các layer sau:
        1. Gọi Backbone (hiện tại là resnet 34 đã được đóng băng/frozen)
        2. Phép chiếu Backbone Featuremap sang d_model của transformer
        3. Encoder của Transformer
        4. Decoder của Transformer
        5. MLP cho Class và BBox
    """
    def __init__(self, config, num_classes, bg_class_idx):
        super().__init__()
        self.backbone_channels = config['backbone_channels']
        self.d_model = config['d_model']
        self.num_queries = config['num_queries']
        self.num_classes = num_classes
        self.num_decoder_layers = config['decoder_layers']
        self.cls_cost_weight = config['cls_cost_weight']
        self.l1_cost_weight = config['l1_cost_weight']
        self.giou_cost_weight = config['giou_cost_weight']
        self.bg_cls_weight = config['bg_class_weight']
        self.nms_threshold = config['nms_threshold']
        self.bg_class_idx = bg_class_idx
        valid_bg_idx = (self.bg_class_idx == 0 or
                        self.bg_class_idx == (self.num_classes-1))
        assert valid_bg_idx, "Background can only be 0 or num_classes-1"

        self.backbone = nn.Sequential(*list(build_backbone(config).children())[:-2])

        if config['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone_proj = nn.Conv2d(self.backbone_channels, self.d_model,
                                       kernel_size=1)
        self.encoder = TransformerEncoder(num_layers=config['encoder_layers'],
                                          num_heads=config['encoder_attn_heads'],
                                          d_model=config['d_model'],
                                          ff_inner_dim=config['ff_inner_dim'],
                                          dropout_prob=config['dropout_prob'])
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.d_model))
        self.decoder = TransformerDecoder(num_layers=config['decoder_layers'],
                                          num_heads=config['decoder_attn_heads'],
                                          d_model=config['d_model'],
                                          ff_inner_dim=config['ff_inner_dim'],
                                          dropout_prob=config['dropout_prob'])
        self.class_mlp = nn.Linear(self.d_model, self.num_classes)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 4),
        )

    def forward(self, x, targets=None, score_thresh=0, use_nms=False):
        # x -> (B, C, H, W)
        # mặc định d_model - 256
        # mặc định C - 3
        # mặc định H, W - 640, 640
        # mặc định feat_h, feat_w - 20, 20

        conv_out = self.backbone(x)  # (B, C_back, feat_h, feat_w)
        # mặc định C_back - 512

        conv_out = self.backbone_proj(conv_out)  # (B, d_model, feat_h, feat_w)

        batch_size, d_model, feat_h, feat_w = conv_out.shape
        spatial_pos_embed = get_spatial_position_embedding(self.d_model, conv_out)
        # spatial_pos_embed -> (feat_h * feat_w, d_model)

        conv_out = (conv_out.reshape(batch_size, d_model, feat_h * feat_w).
                    transpose(1, 2))
        # conv_out -> (B, feat_h * feat_w, d_model)

        # Gọi Encoder
        enc_output, enc_attn_weights = self.encoder(conv_out, spatial_pos_embed)
        # enc_output -> (B, feat_h * feat_w, d_model)
        # enc_attn_weights -> (num_encoder_layers, B, feat_h * feat_w, feat_h * feat_w)

        query_objects = torch.zeros_like(self.query_embed.unsqueeze(0).
                                         repeat((batch_size, 1, 1)))
        # query_objects -> (B, num_queries, d_model)

        decoder_outputs = self.decoder(
            query_objects,
            enc_output,
            self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1)),
            spatial_pos_embed)
        query_objects, decoder_attn_weights = decoder_outputs
        # query_objects -> (num_decoder_layers, B, num_queries, d_model)
        # decoder_attn_weights -> (num_decoder_layers, B, num_queries, feat_h * feat_w)

        cls_output = self.class_mlp(query_objects)
        # cls_output -> (num_decoder_layers, B, num_queries, num_classes)
        bbox_output = self.bbox_mlp(query_objects).sigmoid()
        # bbox_output -> (num_decoder_layers, B, num_queries, 4)

        losses = defaultdict(list)
        detections = []
        detr_output = {}

        if self.training:
            num_decoder_layers = self.num_decoder_layers
            # Thực hiện matching cho mỗi decoder layer
            for decoder_idx in range(num_decoder_layers):
                cls_idx_output = cls_output[decoder_idx]
                bbox_idx_output = bbox_output[decoder_idx]
                with torch.no_grad():
                    # Kết hợp tất cả các prediction boxes và xác suất class lại với nhau
                    class_prob = cls_idx_output.reshape((-1, self.num_classes))
                    class_prob = class_prob.softmax(dim=-1)
                    # class_prob -> (B * num_queries, num_classes)

                    pred_boxes = bbox_idx_output.reshape((-1, 4))
                    # pred_boxes -> (B * num_queries, 4)

                    # Kết hợp tất cả các target boxes và labels lại với nhau
                    target_labels = torch.cat([target["labels"] for target in targets])
                    target_boxes = torch.cat([target["boxes"] for target in targets])
                    # len(target_labels) -> số lượng target cho toàn bộ batch
                    # target_boxes -> (số lượng target cho toàn bộ batch, 4)

                    # Chi phí Phân loại (Classification Cost)
                    cost_classification = -class_prob[:, target_labels]
                    # cost_cls -> (B * num_queries, số lượng target cho toàn bộ batch)

                    # DETR dự đoán cx, cy, w, h, chúng ta cần chuyển sang x1y1x2y2 cho giou
                    # Không cần chuyển đổi targets vì chúng đã ở định dạng x1y1x2y2
                    pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
                        pred_boxes,
                        'cxcywh',
                        'xyxy')

                    cost_localization_l1 = torch.cdist(
                        pred_boxes_x1y1x2y2,
                        target_boxes,
                        p=1
                     )
                    # cost_l1 -> (B * num_queries, số lượng target cho toàn bộ batch)

                    cost_localization_giou = -torchvision.ops.generalized_box_iou(
                        pred_boxes_x1y1x2y2,
                        target_boxes
                    )
                    # cost_giou -> (B * num_queries, số lượng target cho toàn bộ batch)
                    total_cost = (self.l1_cost_weight * cost_localization_l1
                                  + self.cls_cost_weight * cost_classification
                                  + self.giou_cost_weight * cost_localization_giou)

                    total_cost = total_cost.reshape(batch_size, self.num_queries, -1).cpu()
                    # total_cost -> (B, num_queries, số lượng target cho toàn bộ batch)

                    num_targets_per_image = [len(target["labels"]) for target in targets]
                    total_cost_per_batch_image = total_cost.split(
                        num_targets_per_image,
                        dim=-1
                    )
                    # total_cost_per_batch_image[0] = (B, num_queries, số lượng target của ảnh thứ 0)
                    # total_cost_per_batch_image[i] = (B, num_queries, số lượng target của ảnh thứ i)

                    match_indices = []
                    for batch_idx in range(batch_size):
                        batch_idx_assignments = linear_sum_assignment(
                            total_cost_per_batch_image[batch_idx][batch_idx]
                        )
                        batch_idx_pred, batch_idx_target = batch_idx_assignments
                        # len(batch_idx_assignment_pred) = số lượng target của ảnh thứ i
                        match_indices.append((torch.as_tensor(batch_idx_pred,
                                                              dtype=torch.int64),
                                              torch.as_tensor(batch_idx_target,
                                                              dtype=torch.int64)))
                        # match_indices -> [
                        #    ([pred_box_a1, ...], [target_box_i1, ...]),
                        #    ([pred_box_a2, ...], [target_box_i2, ...]),
                        #    ... các cặp assignment cho ảnh thứ i của batch
                        #    ]
                # pred_batch_idxs là các chỉ số batch cho mỗi cặp assignment
                pred_batch_idxs = torch.cat([
                    torch.ones_like(pred_idx) * i
                    for i, (pred_idx, _) in enumerate(match_indices)
                ])
                # pred_batch_idxs -> (số lượng target cho toàn bộ batch, )
                # pred_query_idx là các chỉ số prediction box (trong số num_queries)
                # cho mỗi cặp assignment
                pred_query_idx = torch.cat([pred_idx for (pred_idx, _) in match_indices])
                # pred_query_idx -> (số lượng target cho toàn bộ batch, )

                # Đối với tất cả các prediction boxes đã được assign, lấy target label tương ứng
                valid_obj_target_cls = torch.cat([
                    target["labels"][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)
                ])
                # valid_obj_target_cls -> (số lượng target cho toàn bộ batch, )

                # Khởi tạo target class cho tất cả các predicted boxes là background class
                target_classes = torch.full(
                    cls_idx_output.shape[:2],
                    fill_value=self.bg_class_idx,
                    dtype=torch.int64,
                    device=cls_idx_output.device
                )
                # target_classes -> (B, num_queries)

                # Đối với những predicted boxes được assign cho một target nào đó,
                # cập nhật target label của chúng cho phù hợp
                target_classes[(pred_batch_idxs, pred_query_idx)] = valid_obj_target_cls

                # Để đảm bảo model không bị thiên kiến quá mức vào background class
                cls_weights = torch.ones(self.num_classes)
                cls_weights[self.bg_class_idx] = self.bg_cls_weight

                # Tính classification loss
                loss_cls = torch.nn.functional.cross_entropy(
                    cls_idx_output.reshape(-1, self.num_classes),
                    target_classes.reshape(-1),
                    cls_weights.to(cls_idx_output.device))

                # Lấy tọa độ pred box cho tất cả các matched pred boxes
                matched_pred_boxes = bbox_idx_output[pred_batch_idxs, pred_query_idx]
                # matched_pred_boxes -> (số lượng target cho toàn bộ batch, 4)

                # Lấy tọa độ target box cho tất cả các matched target boxes
                target_boxes = torch.cat([
                    target['boxes'][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)],
                    dim=0
                )
                # target_boxes -> (số lượng target cho toàn bộ batch, 4)

                # Chuyển đổi matched pred boxes sang định dạng x1y1x2y2
                matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
                    matched_pred_boxes,
                    'cxcywh',
                    'xyxy'
                )
                # Không cần chuyển đổi target boxes vì chúng đã ở định dạng x1y1x2y2
                # Tính L1 Localization loss
                loss_bbox = torch.nn.functional.l1_loss(
                    matched_pred_boxes_x1y1x2y2,
                    target_boxes,
                    reduction='none')
                loss_bbox = loss_bbox.sum() / matched_pred_boxes.shape[0]

                # Tính GIoU loss
                loss_giou = torchvision.ops.generalized_box_iou_loss(
                    matched_pred_boxes_x1y1x2y2,
                    target_boxes
                )
                loss_giou = loss_giou.sum() / matched_pred_boxes.shape[0]

                losses['classification'].append(loss_cls * self.cls_cost_weight)
                losses['bbox_regression'].append(
                    loss_bbox * self.l1_cost_weight
                    + loss_giou * self.giou_cost_weight
                )
            detr_output['loss'] = losses
        else:
            # Đối với quá trình inference, chúng ta chỉ quan tâm đến output của layer cuối cùng
            cls_output = cls_output[-1]
            bbox_output = bbox_output[-1]
            # cls_output -> (B, num_queries, num_classes)
            # bbox_output -> (B, num_queries, 4)

            prob = torch.nn.functional.softmax(cls_output, -1)

            # Lấy tất cả các query boxes và class foreground tốt nhất của chúng làm label
            if self.bg_class_idx == 0:
                scores, labels = prob[..., 1:].max(-1)
                labels = labels+1
            else:
                scores, labels = prob[..., :-1].max(-1)

            # Chuyển đổi sang định dạng x1y1x2y2
            boxes = torchvision.ops.box_convert(bbox_output,
                                                'cxcywh',
                                                'xyxy')

            for batch_idx in range(boxes.shape[0]):
                scores_idx = scores[batch_idx]
                labels_idx = labels[batch_idx]
                boxes_idx = boxes[batch_idx]

                # Lọc theo điểm số thấp (Low score filtering)
                keep_idxs = scores_idx >= score_thresh
                scores_idx = scores_idx[keep_idxs]
                boxes_idx = boxes_idx[keep_idxs]
                labels_idx = labels_idx[keep_idxs]

                # Lọc theo NMS (NMS filtering)
                if use_nms:
                    keep_idxs = torchvision.ops.batched_nms(
                        boxes_idx,
                        scores_idx,
                        labels_idx,
                        iou_threshold=self.nms_threshold)
                    scores_idx = scores_idx[keep_idxs]
                    boxes_idx = boxes_idx[keep_idxs]
                    labels_idx = labels_idx[keep_idxs]
                detections.append(
                    {
                        "boxes": boxes_idx,
                        "scores": scores_idx,
                        "labels": labels_idx
                        ,
                    }
                )

            detr_output['detections'] = detections
            detr_output['enc_attn'] = enc_attn_weights
            detr_output['dec_attn'] = decoder_attn_weights
        return detr_output