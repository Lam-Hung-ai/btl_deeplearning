import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
from model.detr import DETR
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Thiết lập device (ưu tiên cuda, sau đó đến cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Kiểm tra nếu có hỗ trợ mps (cho chip Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Đang sử dụng mps')


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Đọc file config #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']

    # Thiết lập seed để đảm bảo khả năng tái lập (reproducibility)
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Khởi tạo dataset và dataloader cho việc train
    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])
    train_dataset = DataLoader(voc,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function)

    # Khởi tạo dataset và dataloader cho việc validation
    val_voc = VOCDataset('val',
                     im_sets=dataset_config['val_im_sets'],
                     im_size=dataset_config['im_size'])
    val_dataset = DataLoader(val_voc,
                               batch_size=train_config['batch_size'],
                               shuffle=False,
                               collate_fn=collate_function)

    # Khởi tạo model và load checkpoint nếu đã tồn tại
    model = DETR(
        config=model_config,
        num_classes=dataset_config['num_classes'],
        bg_class_idx=dataset_config['bg_class_idx']
    )
    model.to(device)
    model.train()

    # Kiểm tra xem file checkpoint có tồn tại hay không
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ckpt_name'])):

        state_dict = torch.load(
            os.path.join(train_config['task_name'],
                         train_config['ckpt_name']),
            map_location=device)
        model.load_state_dict(state_dict)
        print('Đang load checkpoint vì file đã tồn tại')

    # Tạo thư mục task nếu chưa có
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Khởi tạo optimizer AdamW
    optimizer = torch.optim.AdamW(lr=train_config['lr'],
                                  params=filter(lambda p: p.requires_grad,
                                                model.parameters()),
                                  weight_decay=1E-4)

    # Các đoạn comment dưới đây dùng để thiết lập learning rate riêng cho backbone và transformer (nếu cần)
    # backbone_params = [
    #     p for n, p in model.named_parameters() if 'backbone.' in n]
    # transformer_params = [
    #     p for n, p in model.named_parameters() if 'backbone.' not in n]
    # optimizer = torch.optim.AdamW([
    #     {'params': backbone_params, 'lr': train_config['lr']*0.1},
    #     {'params': transformer_params, 'lr': train_config['lr']},
    # ], weight_decay=1e-4)

    # Thiết lập lr_scheduler để giảm learning rate theo các mốc (milestones)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=train_config['lr_steps'],
                               gamma=0.1)
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    
    # Early Stopping parameters
    best_val_loss = float('inf')
    patience = train_config.get('patience', 7)
    trigger_times = 0
    log_file_path = os.path.join(train_config['task_name'], 'training_log.txt')

    # Vòng lặp training qua từng epoch
    for i in range(num_epochs):
        detr_classification_losses = []
        detr_localization_losses = []
        for idx, (ims, targets, _) in enumerate(tqdm(train_dataset)):
            # Chuyển dữ liệu target sang đúng device và format
            for target in targets:
                target['boxes'] = target['boxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
            images = torch.stack([im.float().to(device) for im in ims], dim=0)
            
            # Forward pass và tính loss
            batch_losses = model(images, targets)['loss']

            loss = (sum(batch_losses['classification']) +
                    sum(batch_losses['bbox_regression']))

            detr_classification_losses.append(sum(batch_losses['classification']).item())
            detr_localization_losses.append(sum(batch_losses['bbox_regression']).item())
            
            # Tính toán gradient với kỹ thuật gradient accumulation
            loss = loss / acc_steps
            loss.backward()

            # Cập nhật trọng số model sau một số bước tích lũy (acc_steps)
            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Log kết quả loss định kỳ
            if steps % train_config['log_steps'] == 0:
                loss_output = ''
                loss_output += 'DETR Classification Loss : {:.4f}'.format(
                    np.mean(detr_classification_losses))
                loss_output += ' | DETR Localization Loss : {:.4f}'.format(
                    np.mean(detr_localization_losses))
                print(loss_output, lr_scheduler.get_last_lr())
            
            # Kiểm tra nếu loss bị nan (Not a Number)
            if torch.isnan(loss):
                print('Loss đang trở thành nan. Đang thoát chương trình')
                exit(0)
            steps += 1
            
        # Cập nhật optimizer và scheduler sau mỗi epoch
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        
        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for ims, targets, _ in tqdm(val_dataset, desc='Validation'):
                for target in targets:
                    target['boxes'] = target['boxes'].float().to(device)
                    target['labels'] = target['labels'].long().to(device)
                images = torch.stack([im.float().to(device) for im in ims], dim=0)
                
                batch_losses = model(images, targets)['loss']
                loss = (sum(batch_losses['classification']) +
                        sum(batch_losses['bbox_regression']))
                val_losses.append(loss.item())
        
        model.train()

        avg_train_loss = np.mean(detr_classification_losses) + np.mean(detr_localization_losses)
        avg_val_loss = np.mean(val_losses)

        print('Đã hoàn thành epoch {}'.format(i+1))
        loss_output = ''
        loss_output += 'Train Loss : {:.4f}'.format(avg_train_loss)
        loss_output += ' | Val Loss : {:.4f}'.format(avg_val_loss)
        print(loss_output)
        
        # Log to file
        with open(log_file_path, 'a') as f:
            f.write(f"Epoch {i+1} | {loss_output}\n")
        
        # Lưu model state_dict sau mỗi epoch
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                   train_config['ckpt_name']))

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(train_config['task_name'], 'best_' + train_config['ckpt_name']))
            print("Saved best model.")
        else:
            trigger_times += 1
            print(f'Early stopping trigger times: {trigger_times}/{patience}')
            if trigger_times >= patience:
                print('Early stopping!')
                break
    print('Hoàn tất Training...')


if __name__ == '__main__':
    # Thiết lập các argument dòng lệnh
    parser = argparse.ArgumentParser(description='Arguments cho việc training detr')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)