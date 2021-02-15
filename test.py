import torch, torchvision
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('new_gen_state_dicts/modern_model.pt'))
model.eval()
inp = Image.open('outer/input3.jpg')
comp = transforms.Compose([transforms.ToTensor()])
pred = model([comp(inp)])
print(pred)


def draw_empty_rect(do, xy):
    do.line((xy[0], xy[1], xy[2], xy[1]), fill='red', width=2)
    do.line((xy[2], xy[1], xy[2], xy[3]), fill='red', width=2)
    do.line((xy[0], xy[3], xy[2], xy[3]), fill='red', width=2)
    do.line((xy[0], xy[1], xy[0], xy[3]), fill='red', width=2)


plt.figure()
draw = ImageDraw.Draw(inp)
for k in range(len(pred[0]['boxes'])):
    if pred[0]['scores'][k] >= 0.8:
        box = pred[0]['boxes'][k]
        draw_empty_rect(draw, box)
plt.imshow(inp)
plt.show()