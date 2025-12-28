import torch
import argparse
import os
import yaml
import random
from tqdm import tqdm
from model.detr import DETR
import numpy as np
import cv2
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader

# Kiểm tra thiết bị xử lý (cuda, mps hoặc cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Sử dụng mps')


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area', difficult=None):
    # Cấu trúc của det_boxes:
    # det_boxes = [
    #    {
    #        'person' : [[x1, y1, x2, y2, score], ...],
    #        'car' : [[x1, y1, x2, y2, score], ...]
    #    }
    #    {det_boxes_img_2},
    #    ...
    #    {det_boxes_img_N},
    # ]
    #
    # Cấu trúc của gt_boxes:
    # gt_boxes = [
    #    {
    #        'person' : [[x1, y1, x2, y2], ...],
    #        'car' : [[x1, y1, x2, y2], ...]
    #    },
    #    {gt_boxes_img_2},
    #    ...
    #    {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    all_aps = {}
    # average precisions cho TẤT CẢ các lớp (classes)
    aps = []
    for idx, label in enumerate(gt_labels):
        # Lấy các kết quả dự đoán (detections) của lớp này
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # Cấu trúc cls_dets:
        # cls_dets = [
        #    (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #    ...
        #    (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #    (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #    ...
        #    (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #    ...
        # ]

        # Sắp xếp chúng theo confidence score giảm dần
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # Để theo dõi những gt boxes nào của lớp này đã được khớp (matched)
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Số lượng gt boxes cho lớp này để tính toán recall
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        num_difficults = sum([sum(difficults_label[label])
                              for difficults_label in difficult])

        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # Đối với mỗi dự đoán (prediction)
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Lấy các gt boxes cho hình ảnh này và nhãn này
            im_gts = gt_boxes[im_idx][label]
            im_gt_difficults = difficult[im_idx][label]

            max_iou_found = -1
            max_iou_gt_idx = -1

            # Lấy gt box khớp nhất (best matching)
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            
            # TP chỉ khi iou >= threshold và gt này chưa được khớp trước đó
            if max_iou_found >= iou_threshold:
                if not gt_matched[im_idx][max_iou_gt_idx]:
                    # Nếu là tp thì đánh dấu gt box này đã được khớp
                    gt_matched[im_idx][max_iou_gt_idx] = True
                    tp[det_idx] = 1
                else:
                    fp[det_idx] = 1
            else:
                fp[det_idx] = 1

        # Tính tp và fp cộng dồn (cumulative)
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        # recalls = tp / np.maximum(num_gts, eps)
        recalls = tp / np.maximum(num_gts - num_difficults, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Thay thế các giá trị precision tại recall r bằng giá trị precision tối đa
            # của bất kỳ giá trị recall nào >= r.
            # Điều này dùng để tính toán "precision envelope".
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # Để tính diện tích (area), lấy các điểm mà recall thay đổi giá trị
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Cộng dồn diện tích các hình chữ nhật để tính ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Lấy các giá trị precision cho các giá trị recall >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Lấy giá trị lớn nhất trong các giá trị precision đó
                prec_interp_pt= prec_interp_pt.max() if prec_interp_pt.size>0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method chỉ có thể là area hoặc interp')
        
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
            
    # Tính toán mAP tại iou threshold đã cung cấp
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps


def load_model_and_dataset(args):
    # Đọc file cấu hình (config file) #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    voc = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'])
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)

    model = DETR(
        config=model_config,
        num_classes=dataset_config['num_classes'],
        bg_class_idx=dataset_config['bg_class_idx']
    )
    model.to(device=torch.device(device))
    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['ckpt_name'])), \
        "Không tồn tại checkpoint tại {}".format(os.path.join(train_config['task_name'],
                                                         train_config['ckpt_name']))
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                       train_config['ckpt_name']),
                                     map_location=device))
    return model, voc, test_dataset, config


def infer(args):
    if not os.path.exists('samples'):
        os.mkdir('samples')

    model, voc, test_dataset, config = load_model_and_dataset(args)
    import cv2
    num_samples = 5
    for i in tqdm(range(num_samples)):
        dataset_idx = random.randint(0, len(voc))
        im_tensor, target, fname = voc[dataset_idx]
        detr_output = model(
            im_tensor.unsqueeze(0).to(device),
            score_thresh=config['train_params']['infer_score_threshold'],
            use_nms=config['train_params']['use_nms_infer']
        )
        detr_detections = detr_output['detections']
        enc_attn_weights = detr_output['enc_attn']
        dec_attn_weights = detr_output['dec_attn']

        gt_im = cv2.imread(fname)
        h, w = gt_im.shape[:2]
        gt_im_copy = gt_im.copy()
        
        # Lưu hình ảnh với các ground truth boxes
        for idx, box in enumerate(target['boxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(w*x1), int(h*y1), int(w*x2), int(h*y2)
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = voc.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=voc.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('samples/output_detr_gt_{}.png'.format(i), gt_im)

        # Lấy các dự đoán (predictions) từ model đã train
        boxes = detr_detections[0]['boxes']
        labels = detr_detections[0]['labels']
        scores = detr_detections[0]['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Lưu hình ảnh với các dự đoán (predicted boxes)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(w*x1), int(h*y1), int(w*x2), int(h*y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(voc.idx2label[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite('samples/output_detr_{}.jpg'.format(i), im)
    print('Hoàn tất Detecting...')


def evaluate_map(args):
    model, voc, test_dataset, config = load_model_and_dataset(args)

    gts = []
    preds = []
    difficults = []
    for im_tensor, target, fname in tqdm(test_dataset):
        im_tensor = im_tensor.float().to(device)
        target_bboxes = target['boxes'].float()[0].to(device)
        target_labels = target['labels'].long()[0].to(device)
        difficult = target['difficult'].long()[0].to(device)
        detr_output = model(
            im_tensor,
            score_thresh=config['train_params']['eval_score_threshold'],
            use_nms=config['train_params']['use_nms_eval']
        )
        detr_detections = detr_output['detections']

        boxes = detr_detections[0]['boxes']
        labels = detr_detections[0]['labels']
        scores = detr_detections[0]['scores']

        pred_boxes = {}
        gt_boxes = {}
        difficult_boxes = {}

        for label_name in voc.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []
            difficult_boxes[label_name] = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box in enumerate(target_bboxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            gt_boxes[label_name].append([x1, y1, x2, y2])
            difficult_boxes[label_name].append(difficult[idx].detach().cpu().item())

        gts.append(gt_boxes)
        preds.append(pred_boxes)
        difficults.append(difficult_boxes)

    mean_ap, all_aps = compute_map(preds, gts, method='area', difficult=difficults)
    print('Average Precisions theo từng lớp (Class Wise)')
    for idx in range(len(voc.idx2label)):
        print('AP cho lớp {} = {:.4f}'.format(voc.idx2label[idx],
                                                all_aps[voc.idx2label[idx]]))
    print('Mean Average Precision (mAP) : {:.4f}'.format(mean_ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tham số cho detr inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=True, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    args = parser.parse_args()

    with torch.no_grad():
        if args.infer_samples:
            infer(args)
        else:
            print('Không thực hiện Inference cho mẫu vì đối số `infer_samples` là False')

        if args.evaluate:
            evaluate_map(args)
        else:
            print('Không thực hiện Evaluate vì đối số `evaluate` là False')