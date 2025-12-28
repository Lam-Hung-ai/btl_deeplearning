from dataset.voc import VOCDataset
import yaml

config = None
with open("/home/lamhung/code/btl_deeplearning/config/voc.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
assert config is not None, "Không thể tải file config"
train = VOCDataset('train',
                   im_sets=config['train_im_sets'],
                   im_size=640)
print("Số lượng hình ảnh trong tập train:", len(train))
val = VOCDataset('val',
                 im_sets=config['val_im_sets'],
                 im_size=640)
print("Số lượng hình ảnh trong tập val:", len(val))
test = VOCDataset('test',
                  im_sets=config['test_im_sets'],
                  im_size=640)  
print("Số lượng hình ảnh trong tập test:", len(test))