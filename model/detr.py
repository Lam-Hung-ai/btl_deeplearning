from collections import defaultdict
import torch
import torch.nn as nn
from torchvision.models import resnet34
import torchvision
from scipy.optimize import linear_sum_assignment


def get_spatial_position_embedding(pos_emb_dim, feat_map):
    assert pos_emb_dim % 4 == 1, "Chiều của embedding vị trí phải chia hết cho 4"
    
    # Lấy H, W từ feature map (Giả sử đầu vào là B, C, H, W)
    h, w = feat_map.shape[2], feat_map.shape[3]
    device = feat_map.device

    # 1. Tạo lưới tọa độ
    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    
    # grid_y, grid_x shape: (H, W)
    grid_y, grid_x = torch.meshgrid(grid_h, grid_w, indexing="ij")
    
    # Phẳng hóa để tính toán: (H*W,)
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    # 2. Tính toán hệ số tần số (denominator)
    # d_half = pos_emb_dim // 2
    dim_t = torch.arange(pos_emb_dim // 4, dtype=torch.float32, device=device)
    factor = 10000 ** (dim_t / (pos_emb_dim // 4))

    # 3. Tính Embedding cho Hàng (Y) và Cột (X)
    # Sử dụng broadcasting thay vì .repeat()
    emb_y = grid_y[:, None] / factor  # (H*W, pos_emb_dim // 4)
    emb_x = grid_x[:, None] / factor  # (H*W, pos_emb_dim // 4)

    # Tạo vector [Sin, Cos] cho mỗi trục
    grid_h_emb = torch.cat([torch.sin(emb_y), torch.cos(emb_y)], dim=-1)
    grid_w_emb = torch.cat([torch.sin(emb_x), torch.cos(emb_x)], dim=-1)

    # 4. Nối Hàng và Cột để có vector cuối cùng
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1) # (H*W, pos_emb_dim)

    return pos_emb



class TransformerEncoder(nn.Module):
    """
    1. LayerNorm cho self attention
    2. Self attention
    3. LayerNorm cho MLP
    4. MLP
    """

    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim, dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Self Attention Module cho tất cả encoder layer
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=self.dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        # MLP Module cho tất cả lớp encoder
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model)
                )
                for _ in range(num_layers)
            ]
        )

        # Chuẩn hóa cho attention tất cả lớp encoder
        self.attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ]
        )

        # Chuẩn hóa cho MLP cho tất cả lớp encoder
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ]
        )

        # Dropout cho self attention cho tất cả lớp encoder 
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ]
        )

        # Dropout MLP cho tất cả lớp encoder
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ]
        )

        # Chuẩn hóa cho đầu ra encoder
        self.output_norm = nn.LayerNorm(d_model)


    def forward(self, x, spatial_position_embedding):
        out = x
        attn_weights = []
        for i in range(self.num_layers):

            # Norm, Self Attention, Dropout và Residual
            in_attn = self.attn_norms[i](out)
            # Thêm spatial position embedding
            # Thêm q, k cho tất cả self attention
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
            out = out_ff + in_ff

        # Chuẩn hóa đầu ra (Output normalization)
        out = self.output_norm(out)
        return out, torch.stack(attn_weights)


class TransformerDecoder(nn.Module):
    """
    Mỗi lớp TransformerDecoder có những module này:
        1. LayerNorm cho self attention
        2. Self attention
        3. LayerNorm cho Cross attention cho đầu ra của encoder
        4. Cross Attention
        5. LayerNorm cho MLP
        6. MLP
    """

    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim, dropout_prob = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout_prob

        # Self Attention module cho tất cả lớp decoder
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        # Cross Attention Module cho tất cả lớp decoder
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        # MLP module cho tất cả lớp decoder
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(ff_inner_dim, d_model)
                )
                for _ in range(num_layers)
            ]
        )

        # Chuẩn hóa tất cả attention module cho tất cả decoder
        self.attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ]
        )

        # Chuẩn hóa cho Cross Attention cho tất cả lớp decoder
        self.cross_attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ]
        )

        # Chuẩn hóa cho lớp MLP cho cả lớp decoder
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ]
        )

        # Dropout tất cả attention module cho tất cả decoder
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(dropout_prob)
                for _ in range(num_layers)
            ]
        )

        # Dropout tất cả Cross attention module cho tất cả decoder
        self.cross_attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(dropout_prob)
                for _ in range(num_layers)
            ]
        )
        
        # Dropout tất cả MLP module cho tất cả decoder
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(dropout_prob)
                for _ in range(num_layers)
            ]
        )

        # Chia sẻ output norm cho toàn bộ lớp decoder
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_objects, encoder_output, query_embedding, spatial_position_embedding):
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []
        for i in range(self.num_layers):
            # Norm, attention, dropout và residual
            in_attn = self.attn_norms[i](out)
            q = in_attn + query_embedding
            k = in_attn + query_embedding
            out_attn, _ = self.attns[i](
                query=q,
                key=k,
                value=in_attn
            )
            out_attn = self.attn_dropouts[i](out_attn)
            out = out_attn + out

            # Norm, cross attention, dropout và residual
            in_attn = self.cross_attn_norms[i](out)
            q = in_attn + query_embedding
            k = encoder_output + spatial_position_embedding
            out_attn, decoder_cross_attn = self.cross_attns[i](
                query = q,
                key=k,
                value = in_attn
            )
            decoder_cross_attn_weights.append(decoder_cross_attn)
            out_attn = self.cross_attn_dropouts[i](out_attn)
            out = out_attn + out

            # Norm, MLP, dropout và residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out_ff + out
            decoder_outputs.append(self.output_norm(out))

        output = torch.stack(decoder_outputs)
        
        return output, torch.stack(decoder_cross_attn_weights)

class DETR(nn.Module):
    """
    Mô hình DETR là sự kết hợp giữa CNN + Transformer + thuật toán Hungarian

    Các bước forward:
        1. Backbone CNN (resnet34)
        2. Chiếu feature map của backbone sang d_model của transformer
        3. Encoder của Transformer
        4. Decoder của Transformer
        5. Lớp phân loại (Class) và dự đoán BBox từ MLP
    """

    def __init__(self, config, num_classes, bg_class_idx):
        super().__init__()
        self.backbone_channels = config["backbone_channels"]
        self.d_model = config["d_model"]
        self.num_queries = config["num_queries"]
        self.num_classes = num_classes
        self.num_decode_layers  = config["decoder_layers"]
        self.cls_cost_weight = config["cls_cost_weight"]
        self.l1_cost_weight = config["l1_cost_weight"]
        self.giou_cost_weight = config["giou_cost_weight"]
        self.bg_cls_weight = config["bg_class_weight"]
        self.nms_threshold = config["nms_threshold"]
        self.bg_class_idx = bg_class_idx
        valid_bg_idx = (self.bg_class_idx == 0 or self.bg_class_idx == self.num_classes-1)
        assert valid_bg_idx, "Chỉ số cho nhãn Background chỉ có thể bằng 0 hoặc num_classes-1"

        self.backbone = nn.Sequential(*list(
            resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1, norm_layer=torchvision.ops.FrozenBatchNorm2d).children()
        )[:-2])

        if config["freeze_backbone"]:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone_proj = nn.Conv2d(self.backbone_channels, self.d_model, kernel_size=1)

        self.encoder = TransformerEncoder(num_layers=config['encoder_layers'],
                                          num_heads=config["encoder_attn_heads"],
                                          d_model = config["d_model"],
                                          ff_inner_dim=config["ff_inner_dim"],
                                          dropout_prob=config["dropout_prob"])
        self.query_embeded = nn.Parameter(torch.randn(self.num_queries, self.d_model))
        self.decoder = TransformerDecoder(num_layers=config['encoder_layers'],
                                          num_heads=config["encoder_attn_heads"],
                                          d_model = config["d_model"],
                                          ff_inner_dim=config["ff_inner_dim"],
                                          dropout_prob=config["dropout_prob"])
        self.class_mlp = nn.Linear(self.d_model, self.num_classes)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, out_features=4)
        )

    def forward(self, x, targets=None, score_thresh=0, use_nms=False):
        # x -> (B, C, H, W)
        # mặc định d_model = 256
        # mặc định C = 3
        # mặc định H, W = 640, 640
        # mặc định feat_h, feat_w = 20, 20

        conv_out = self.backbone(x) # (B, C_back, feat_h, feat_w)
        # mặc định C_back = 512

        conv_out = self.backbone_proj(conv_out) # (B, d_model, feat_h, feat_w)

        batch_size, d_model, feat_h, feat_w = conv_out.shape
        spatial_pos_embed = get_spatial_position_embedding(self.d_model, conv_out)
        # spatial_pos_embed -> (feat_h * feat_w, d_model)

        conv_out = (conv_out.reshape(batch_size, d_model, feat_h*feat_w).transpose(1, 2))
        # conv_out -> (B, feat_h * feat_w, d_model)

        # encoder 
        enc_output, enc_attn_weights = self.encoder(conv_out, spatial_pos_embed)
        # enc_output -> (B, feat_h * feat_w, d_model)
        # enc_attn_weights -> (num_encoder_layers, B, feat_h * feat_w, feat_h * feat_w)

        query_objects = torch.zeros_like(self.query_embeded.unsqueeze(0).repeat((batch_size, 1, 1)))
        # query_objects -> (B, num_queries, d_model)

        decoder_outputs = self.decoder(
            query_objects,
            enc_output,
            self.query_embeded.unsqueeze(0).repeat((batch_size, 1, 1)),
            spatial_pos_embed
        )
        query_objects, decoder_attn_weights = decoder_outputs
        # query_objects -> (num_decoder_layers, B, num_queries, d_model)
        # decoder_attn_weights -> (num_decoder_layers, B, num_queries, feat_h*feat_w)

        cls_output = self.class_mlp(query_objects)
        # cls_output -> (num_decoder_layers, B, num_queries, num_classes)
        bbox_output = self.bbox_mlp(query_objects).sigmoid()
        # bbox_output -> (num_decoder_layers, B, num_queries, 4)

        losses = defaultdict(list)
        detections = []
        detr_output = {}

        if self.training:
            num_decoder_layers = self.num_decode_layers
            for decoder_idx in range(num_decoder_layers):
                cls_idx_output = cls_output[decoder_idx]
                bbox_idx_output = bbox_output[decoder_idx]
                with torch.no_grad():
                    # Nối tất cả bbox dự đoán và xác suất lớp lại với nhau
                    class_prob = cls_idx_output.reshape((-1, self.num_classes))
                    class_prob = class_prob.softmax(dim=-1)
                    # class_prob -> (B*num_queries, num_classes)

                    pred_boxes = bbox_idx_output.reshape((-1, 4))
                    # pred_boxes -> (B*num_queries, 4)

                    # Nối tất cả các nhãn thực (ground truth labels) và hộp thực (boxes) lại với nhau
                    assert targets is not None, "Cần nhãn thực để huấn luyện"
                    target_labels = torch.cat([target["labels"] for target in targets])
                    target_boxes = torch.cat([target["boxes"] for target in targets])
                    # len(target_labels) -> num_targets_for_entire_batch
                    # target_boxes -> (num_targets_for_entire_batch, 4)

                    # Chi phí Phân loại (Classification Cost)
                    cost_classification = -class_prob[:, target_labels]
                    # cost_classification -> (B*num_queries, num_targets_for_entire_batch)

                    # DETR dự đoán cx, cy, w, h vì vậy chúng ta cần chuyển về x1y1x2y2 cho GIoU
                    # Không cần chuyển đổi targets bởi vì nó đã có dạng x1y1x2y2
                    pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
                        pred_boxes,
                        "cxcywh",
                        "xyxy"
                    )

                    cost_localization_l1 = torch.cdist(
                        pred_boxes_x1y1x2y2,
                        target_boxes,
                        p=1
                    )

                    cost_localization_giou = -torchvision.ops.generalized_box_iou(
                        pred_boxes_x1y1x2y2,
                        target_boxes
                    )
                    # cost_giou -> (B*num_queries, num_targets_for_entire_batch)

                    total_cost = self.l1_cost_weight * cost_localization_l1 + self.cls_cost_weight * cost_classification + self.giou_cost_weight * cost_localization_giou
                                                      
                    total_cost = total_cost.reshape(batch_size, self.num_queries, -1).cpu()
                    # total_cost -> (B, num_queries, num_targets_for_entire_batch)

                    num_targets_per_image = [len(target["labels"]) for target in targets]
                    total_cost_per_batch_image = total_cost.split(
                        num_targets_per_image,
                        dim=-1
                    )
                    # total_cost_per_batch_image[0] = (B, num_queries, num_targets_0th_image)
                    # total_cost_per_batch_image[1] = (B, num_queries, num_targets_1th_image)

                    match_indices = []
                    for batch_idx in range(batch_size):
                        batch_idx_assignments = linear_sum_assignment(
                            total_cost_per_batch_image[batch_idx][batch_idx]
                        )

                        batch_idx_pred, batch_idx_target = batch_idx_assignments
                        # len(batch_idx_assignments_pred) = num_targets_ith_image

                        match_indices.append(
                            (torch.as_tensor(batch_idx_pred, dtype=torch.int64),
                            torch.as_tensor(batch_idx_target, dtype=torch.int64))
                            )
                        # match_indices -> [
                        #     ([pred_box_a1, ...], [target_box_i1, ...]),
                        #     ([pred_box_a2, ...], [target_box_i2, ...]),
                        #     ... cặp gán (assignment) cho ảnh thứ i trong batch
                        # ]
                        

                # pred_batch_idxs là các chỉ số batch cho mỗi cặp gán
                pred_batch_idxs = torch.cat([
                    torch.ones_like(pred_idx) * i
                    for i, (pred_idx, _) in enumerate(match_indices)
                ])
                # pred_batch_idxs -> (num_targets_for_entire_batch, )
                
                # pred_query_idx là chỉ số của prediction box (trong tổng số num_queries) cho mỗi cặp gán
                pred_query_idx = torch.cat([pred_idx for (pred_idx, _) in match_indices])
                # pred_query_idx -> (num_targets_for_entire_batch, )

                # Đối với tất cả các prediction boxes đã được gán, lấy nhãn thực (target label)
                valid_obj_target_cls = torch.cat([
                    target["labels"][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)
                ])
                # valid_obj_target_cls -> (num_targets_for_entire_batch, )

                # Khởi tạo nhãn mục tiêu cho tất cả các predicted boxes là lớp nền (background class)
                target_classes = torch.full(
                    cls_idx_output.shape[:2],
                    fill_value=self.bg_class_idx,
                    dtype=torch.int64,
                    device=cls_idx_output.device
                )
                # target_classes -> (B, num_queries)

                # Đối với các predicted boxes đã được gán cho một mục tiêu nào đó,
                # cập nhật nhãn mục tiêu tương ứng
                target_classes[(pred_batch_idxs, pred_query_idx)] = valid_obj_target_cls

                # Đảm bảo lớp nền không chiếm tỷ trọng quá lớn khi mô hình tính toán (balancing)
                cls_weights = torch.ones(self.num_classes)
                cls_weights[self.bg_class_idx] = self.bg_cls_weight

                # Tính toán Classification loss
                loss_cls = torch.nn.functional.cross_entropy(
                    cls_idx_output.reshape(-1, self.num_classes),
                    target_classes.reshape(-1),
                    cls_weights.to(cls_idx_output.device))

                # Lấy tọa độ pred box cho tất cả các pred box đã được khớp (matched)
                matched_pred_boxes = bbox_idx_output[pred_batch_idxs, pred_query_idx]
                # matched_pred_boxes -> (num_targets_for_entire_batch, 4)

                # Lấy tọa độ target box cho tất cả các target boxes đã được khớp
                target_boxes = torch.cat([
                    target['boxes'][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)],
                    dim=0
                )
                # target_boxes -> (num_targets_for_entire_batch, 4)

                # Chuyển đổi matched pred boxes sang định dạng x1y1x2y2
                matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
                    matched_pred_boxes,
                    'cxcywh',
                    'xyxy'
                )
                
                # Không cần chuyển đổi target boxes vì chúng đã ở định dạng x1y1x2y2
                # Tính toán L1 Localization loss
                loss_bbox = torch.nn.functional.l1_loss(
                    matched_pred_boxes_x1y1x2y2,
                    target_boxes,
                    reduction='none')
                loss_bbox = loss_bbox.sum() / matched_pred_boxes.shape[0]

                # Tính toán GIoU loss
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
            # Đối với inference, chúng ta chỉ quan tâm đến đầu ra của layer cuối cùng
            cls_output = cls_output[-1]
            bbox_output = bbox_output[-1]
            # cls_output -> (B, num_queries, num_classes)
            # bbox_output -> (B, num_queries, 4)

            prob = torch.nn.functional.softmax(cls_output, -1)

            # Lấy tất cả các query boxes và lớp tiền cảnh (foreground class) tốt nhất làm nhãn
            if self.bg_class_idx == 0:
                scores, labels = prob[..., 1:].max(-1)
                labels = labels+1
            else:
                scores, labels = prob[..., :-1].max(-1)

            # Chuyển sang định dạng x1y1x2y2
            boxes = torchvision.ops.box_convert(bbox_output,
                                                'cxcywh',
                                                'xyxy')

            for batch_idx in range(boxes.shape[0]):
                scores_idx = scores[batch_idx]
                labels_idx = labels[batch_idx]
                boxes_idx = boxes[batch_idx]

                # Lọc theo ngưỡng điểm số thấp (Low score filtering)
                keep_idxs = scores_idx >= score_thresh
                scores_idx = scores_idx[keep_idxs]
                boxes_idx = boxes_idx[keep_idxs]
                labels_idx = labels_idx[keep_idxs]

                # Lọc bằng NMS (Non-Maximum Suppression)
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