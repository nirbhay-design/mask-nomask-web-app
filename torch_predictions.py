import torch
import torchvision
from PIL import Image

transformations = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()
])

def load_img():
    img_path = "static/uploaded_imgs/Inferenceimage.png"
    img = Image.open(img_path).convert('RGB')
    img = transformations(img)
    img = img.unsqueeze(0)
    return img

def infer(img):
    mob_v2 = torchvision.models.mobilenet_v2(pretrained=True)
    mob_v2.classifier[0] = torch.nn.Dropout(p=0.3)
    mob_v2.classifier[1] = torch.nn.Linear(in_features=1280,out_features=2)

    checkpoint = torch.load('torch-model/mask_classification_v1.pth',map_location='cpu')
    mob_v2.load_state_dict(checkpoint['model'])
    mob_v2.eval()

    scores = mob_v2(img)
    softmax_scores = torch.nn.functional.softmax(scores,dim=1)
    _,predictions = softmax_scores.max(dim=1)
    inference = "masked" if predictions[0].item() == 0 else "unmasked"
    return inference

