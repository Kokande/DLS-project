import torch, torchvision
from torchvision.transforms import transforms

import numpy as np
import os
from PIL import Image, ImageDraw
from copy import deepcopy
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

device = torch.device('cpu')
im = [Image.open("./outer/input1.jpg").convert("RGB"), Image.open("./outer/input2.jpg").convert("RGB"), Image.open("./outer/input3.jpg").convert("RGB")]
for i in im:
    plt.figure()
    plt.imshow(i)
    plt.show()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
frpil = transforms.ToTensor()
pred = model([frpil(i) for i in im])


def draw_empty_rect(do, xy):
    do.line((xy[0], xy[1], xy[2], xy[1]), fill='red', width=2)
    do.line((xy[2], xy[1], xy[2], xy[3]), fill='red', width=2)
    do.line((xy[0], xy[3], xy[2], xy[3]), fill='red', width=2)
    do.line((xy[0], xy[1], xy[0], xy[3]), fill='red', width=2)


for i in range(3):
    plt.figure()
    draw = ImageDraw.Draw(im[i])
    for k in range(len(pred[i]['boxes'])):
        if pred[i]['scores'][k] >= 0.9:
            box = pred[i]['boxes'][k]
            draw_empty_rect(draw, box)
    plt.imshow(im[i])
    plt.show()


def find_area(xy):
    return (float(xy[2]) - float(xy[0])) * (float(xy[3]) - float(xy[1]))


class WiderFaceDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        with open(txt_file) as f:
            lines = f.readlines()
            self.dataset = []
            id = 0
            for ind, line in enumerate(lines):
                if line.rstrip().endswith('.jpg'):
                    self.dataset.append({'file_name': line.rstrip(),
                                         'targets': {'boxes': [],
                                                     'image_id': torch.tensor([id]),
                                                     'labels': [],
                                                     'area': [],
                                                     'iscrowd': []}})
                    id += 1
                elif len(line.split()) != 1:
                    inp = line.split()
                    self.dataset[-1]['targets']['boxes'].append(np.array([float(inp[0]),
                                                                          float(inp[1]),
                                                                          float(inp[0]) + float(inp[2]),
                                                                          float(inp[1]) + float(inp[3])]))
                    self.dataset[-1]['targets']['labels'].append(1)
                    if bool(self.dataset[-1]['targets']['boxes']):
                        self.dataset[-1]['targets']['area'].append(find_area(self.dataset[-1]['targets']['boxes'][-1]))
                    self.dataset[-1]['targets']['iscrowd'].append(False)
        self.dataset = self.dataset[:200]
        self.root_dir = root_dir
        self.transform = transform
 
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.dataset[idx]['file_name'])
        image = Image.open(img_name).convert('RGB')
        targets = self.dataset[idx]['targets']
        sample = [image,
                  {'boxes': torch.tensor(targets['boxes']),
                   'image_id': targets['image_id'],
                   'labels': torch.tensor(targets['labels'], device=device).type(torch.int64),
                   'area': torch.tensor(targets['area']),
                   'iscrowd': torch.tensor(targets['iscrowd'])}]

        if self.transform:
            sample = self.transform(sample)
 
        return sample


class MyDataLoader(object):
    def __init__(self, ds, batch_size, shuffle=False):
        self.dataset = ds
        self.batch_size = batch_size
        if shuffle:
            self.perm = np.random.permutation(len(ds))
        else:
            self.perm = np.arange(len(ds))
        self.batch_num = len(ds) // batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            features = []
            labels = []
            if i + self.batch_size >= len(self.dataset):
                break
            for k in range(i, i + self.batch_size):
                features.append(self.dataset[self.perm[k]][0])
                labels.append({'boxes': self.dataset[self.perm[k]][1]['boxes'],
                               'labels': self.dataset[self.perm[k]][1]['labels'],
                               'image_id': self.dataset[self.perm[k]][1]['image_id'],
                               'area': self.dataset[self.perm[k]][1]['area'],
                               'iscrowd': self.dataset[self.perm[k]][1]['iscrowd']})
            print('iter_passed')
            yield [features, labels]

    def __len__(self):
        return self.batch_num


class MyToTensor(object):
    def __call__(self, sample):
        tf = transforms.ToTensor()
        sample[0] = tf(sample[0])
        return sample


class MyResize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.resize = transforms.Resize(output_size)

    def __call__(self, sample):
        image, targets = sample[0], sample[1]

        h, w = image.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = self.resize(image)
        boxes = targets['boxes']
        for i in range(len(boxes)):
            boxes[i] = torch.tensor([boxes[i][0] * new_w / w, 
                                     boxes[i][1] * new_h / h,
                                     boxes[i][2] * new_w / w + 1, 
                                     boxes[i][3] * new_h / h + 1])
        targets['boxes'] = boxes

        return [img, targets]


train_set = WiderFaceDataset('WIDER_train/wider_face_train_bbx_gt.txt', 
                             'WIDER_train/images', 
                             transform=transforms.Compose([MyToTensor(),
                                                           MyResize((800, 800))]))
val_set = WiderFaceDataset('WIDER_val/wider_face_val_bbx_gt.txt', 
                           'WIDER_val/images', 
                           transform=transforms.Compose([MyToTensor(),
                                                         MyResize((800, 800))]))


def draw_empty_rect(do, xy):
    do.line((xy[0], xy[1], xy[2], xy[1]), fill='red', width=2)
    do.line((xy[2], xy[1], xy[2], xy[3]), fill='red', width=2)
    do.line((xy[0], xy[3], xy[2], xy[3]), fill='red', width=2)
    do.line((xy[0], xy[1], xy[0], xy[3]), fill='red', width=2)


def show_img(image, targets):
    boxes = targets['boxes']
    tf = transforms.ToPILImage()
    img = tf(deepcopy(image))
    for i in boxes:
        draw = ImageDraw.Draw(img)
        draw_empty_rect(draw, i)
    plt.figure()
    plt.imshow(img)
    plt.show()


sample = train_set[12]
show_img(sample[0], sample[1])

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("older_model4.pt"))

from engine import train_one_epoch, evaluate

train_loader = MyDataLoader(train_set, 10, shuffle=True)
val_loader = MyDataLoader(val_set, 1)
num_classes = 2
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
num_epochs = 10
epochs_models = []
for epoch in range(5, num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
    torch.save(model.state_dict(), f'older_model{epoch}.pt')
    lr_scheduler.step()
    evaluate(model, val_loader, device=device)
torch.save(model.state_dict(), 'modern_model.pt')
model.eval()
toten = transforms.ToTensor()
pred = model([toten(im[2])])
show_img(im[2], pred[0])