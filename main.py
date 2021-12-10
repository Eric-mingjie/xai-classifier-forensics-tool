import os
import os.path
import datetime
import tempfile
import io
from PIL import Image
import base64
import threading
import torchvision.transforms.functional as F
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import time
import numpy as np
import skimage.io 
from skimage import img_as_ubyte

import json
from torchvision import transforms
import traceback
from jinja2 import Template
from IPython.display import HTML, Javascript, JSON, IFrame, display

from flask import Flask
from flask import render_template, request, session, abort, jsonify, send_from_directory
from flask_cors import CORS

from attacks import Attacker, PGD_L2, DDN

import sys
sys.path.append("smoothing/code")
from architectures import get_architecture

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

port=os.environ["PORT"] 

attack_kwargs = {
    'tv': 0,
    'momentum': 0,
    'step_size': 2.0,
    'no_grad': False,
}

model_args_1 = {
    'denoiser_path': "smoothing/denoisers/depth_5_nc_16/imagenet_mse/noise_1.00/checkpoint.pth.tar",
    'denoiser_depth': 5,
    'denoiser_nc': 16,
}

model_args_2 = {
    'denoiser_path': "smoothing/denoisers/dncnn_5epoch_lr1e-4/noise_1.00/checkpoint.pth.tar",
}

model = None
##################################################################
app = Flask(__name__)
CORS(app)

def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]

def generate_adv(model, classifier, batch, attacker, targeted, num_noise_vec, noise_sd, step_size, tv, mask):
    mini_batches = get_minibatches(batch, 1)

    for inputs, targets in mini_batches:
        inputs = inputs.cuda()
        targets = targets.cuda()

        # inputs = inputs.repeat((args.num_noise_vec, 1, 1, 1))
        shape = list(inputs.shape)
        shape[0] = inputs.shape[0] * num_noise_vec
        inputs = inputs.repeat((1, num_noise_vec, 1, 1)).view(shape)
        noise = torch.randn_like(inputs).cuda() * noise_sd

        # Attack the smoothed classifier 
        print("attacking")
        inputs_adv = attacker.attack(model, inputs, targets, noise=noise,
            num_noise_vectors=num_noise_vec,
            targeted=targeted,
            step_size=step_size, 
            tv=tv,
            mask=mask)

        print("attack finished")

        # Compute the predicted class of generated adversarial examples
        with torch.no_grad():
            inputs_adv_noise = inputs_adv # + torch.randn_like(inputs).cuda() * noise_sd
            outputs_adv = classifier(inputs_adv_noise)

        print("generating predictions on the adversarial images")

        out_img_list = []
        out_adv_class_smoothed = []
        out_adv_img_list_np = []
        for i in range(batch[0].shape[0]):
            new_img = inputs_adv[num_noise_vec*i].detach().cpu()
            permute = [2, 1, 0]
            new_img = new_img[permute,:,:]
            out_adv_img_list_np.append(np.transpose(new_img, [1,2,0]))
            new_img_pil = F.to_pil_image(new_img)

            buffered = io.BytesIO()
            new_img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            out_img_list.append(img_str.decode("utf-8"))

            ################################################
            outputs_adv_i = outputs_adv[num_noise_vec*i: num_noise_vec*(i+1)]
            outputs_adv_cls = outputs_adv_i.argmax(1)

            counts_adv = np.zeros(5, dtype=int)
            for idx in outputs_adv_cls:
                counts_adv[idx] += 1.0

            class_adv = counts_adv.argsort()[::-1][0]
            out_adv_class_smoothed.append(str(class_adv))

    return out_adv_class_smoothed, out_img_list, out_adv_img_list_np

@app.route('/static/pics/<id>/<filename>')
def serve_pic(id,filename):
    print("sending images")
    if "TrojAI" in str(id) and "Poisoned" in str(id):
        base = "models/trojai/poisoned/example_data"
    elif "TrojAI" in str(id) and "Clean" in str(id):
        base = "models/trojai/clean/example_data"
    elif "ImageNet" in str(id):
        base = "models/imagenet/example_data"

    return send_from_directory(base, filename+".png")

@app.route('/loading')
def serve_loading():
    return send_from_directory('robust_model_colab/static/', 'loading.gif')

@app.route('/static/css/style.css')
def serve_css():
    return send_from_directory('robust_model_colab/static/css/', 'style.css')

@app.route('/debug')
def serve_debug():
    return "hello world!"

@app.route('/')
def serve_main():
    html_template = Template(open('robust_model_colab/index.html').read())
    jscolor_script = open("robust_model_colab/static/js/jscolor.js").read()
    index_script = Template(open("robust_model_colab/static/js/index.js").read()).render(port=port)
    draw_script = Template(open("robust_model_colab/static/js/draw.js").read()).render(port=port)

    return html_template.render(
                         jscolor_script=jscolor_script,
                         index_script=index_script,
                         draw_script=draw_script,
                         port=port,
                         )

@app.route("/cleanpredict", methods=['POST'])
def cleanpredict():
    start = time.time()
    data = request.get_json(force=True)

    clf = str(data["clf"])

    #############################################################################################
    # clf_path = "/project_data/datasets/trojai/trojai-round0-dataset/id-%.8d/model.pt"%clf_id
    if "TrojAI" in clf and "Poisoned" in clf:
        clf_path = "models/trojai/poisoned/model.pt"
    elif "TrojAI" in clf and "Clean" in clf:
        clf_path = "models/trojai/clean/model.pt"
    elif "ImageNet" in clf and "Poisoned" in clf:
        clf_path = "models/imagenet/poisoned/poisoned_model.pt"
    elif "ImageNet" in clf and "Clean" in clf:
        clf_path = "models/imagenet/clean/clean_model.pt" 
    ##################################################################

    if "TrojAI" in clf:
        classifier = torch.load(clf_path)
    elif "ImageNet" in clf:
        model_ft = models.alexnet(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,5)
        normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        classifier = nn.Sequential(normalize, model_ft)
        checkpoint = torch.load(clf_path, map_location='cuda:0')
        classifier.load_state_dict(checkpoint["state_dict"])

    model = torch.nn.DataParallel(classifier).cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print("model loaded")
    ##############################################################################################
    if "TrojAI" in clf and "Poisoned" in clf:
        base = "models/trojai/poisoned/example_data"
    elif "TrojAI" in clf and "Clean" in clf:
        base = "models/trojai/clean/example_data"
    elif "ImageNet" in clf:
        base = "models/imagenet/example_data"

    out_img_list = []
    out_adv_class = []

    print("before loading images")

    try:
        img_list = []
        target_list = []

        for class_id in range(5):
            for i in range(5):
                imgpath = os.path.join(base, "class_%d_example_%d.png"%(class_id, i))
                img = skimage.io.imread(imgpath)
                if "TrojAI" in clf:
                    r = img[:, :, 0]
                    g = img[:, :, 1]
                    b = img[:, :, 2]
                    img = np.stack((b, g, r), axis=2)
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0)
                img = img - np.min(img)
                img = img / np.max(img)

                img_list.append(img)
                target_list.append(class_id)

        inputs = torch.cuda.FloatTensor(np.concatenate(img_list, axis=0))
        targets = torch.cuda.LongTensor(np.array(target_list))
        batch = [inputs, targets]

        print("images loaded")
        mini_batches = get_minibatches(batch, 1)

        for inputs, targets in mini_batches:
            inputs = inputs.cuda()
            targets = targets.cuda()

            ##############################################################################
            print("generating prediction on clean images")
            # Compute the predicted class of generated adversarial examples
            with torch.no_grad():
                inputs_noise = inputs #+ noise
                try:
                    outputs = model(inputs_noise)
                except:
                    traceback.print_exc()

            print("clean image prediction finished")
            out_img_list = []
            out_clean_class = []
            # out_clean_class_orig = []
            for i in range(batch[0].shape[0]):
                outputs_clean_i = outputs[i]
                outputs_clean_cls = outputs_clean_i.argmax()

                out_clean_class.append(str(outputs_clean_cls.item()))

        return jsonify({'clean_pred_class': out_clean_class})
    except:
        tb = traceback.format_exc()
        return tb

@app.route("/predict", methods=['POST'])
def predict():
    print("test")
    data = request.get_json(force=True)

    # clf_id = int(data["clf_id"])
    clf = str(data["clf"])
    startX = int(data["startX"])
    startY = int(data["startY"])
    lenX = int(data["lenX"])
    lenY = int(data["lenY"])
    color = data["color"]
    patch_option = data["patch_option"]
    canvas_id_str = data["canvas_id_str"]

    startX_v2 = int(data["startX_v2"])
    startY_v2 = int(data["startY_v2"])

    startX_v2 = max(startX_v2 - int(lenX/2), 0)
    startY_v2 = max(startY_v2 - int(lenY/2), 0)

    color = color.lstrip('#')
    color_rgb = [int(color[i:i+2], 16) for i in (0, 2, 4)]
    print(color_rgb)

    if "cropped" in patch_option:
        img_i, img_j = int(canvas_id_str[-11])-1, int(canvas_id_str[-1])-1

        adv_img = adv_img_list[img_i * 5 + img_j]
        cropped_patch = adv_img[startY:(startY+lenY),startX:(startX+lenX),:]

    print("startX %f startY %f lenX %f lenY %f"%(startX, startY, lenX, lenY))
    #################################################################################
    # base = "/project_data/datasets/trojai/trojai-round0-dataset/id-%.8d/example_data"%clf_id
    if "TrojAI" in clf and "Poisoned" in clf:
        base = "models/trojai/poisoned/example_data"
    elif "TrojAI" in clf and "Clean" in clf:
        base = "models/trojai/clean/example_data"
    elif "ImageNet" in clf:
        base = "models/imagenet/example_data"

    img_list = []
    for class_id in range(5):
        for i in range(5):
            imgpath = os.path.join(base, "class_%d_example_%d.png"%(class_id, i))
            img = skimage.io.imread(imgpath)

            if "color" in patch_option:
                img[startY:startY+lenY, startX:startX+lenX, 0]=color_rgb[0]
                img[startY:startY+lenY, startX:startX+lenX, 1]=color_rgb[1]
                img[startY:startY+lenY, startX:startX+lenX, 2]=color_rgb[2]

            img = img - np.min(img)
            img = img / np.max(img)

            if "cropped" in patch_option:
                # img[100:(100+lenY),100:(100+lenX),:] = cropped_patch
                img[startY_v2:(startY_v2+lenY),startX_v2:(startX_v2+lenX),:] = cropped_patch

            # skimage.io.imsave("vis_results/debug/c_%d_e_%d.png"%(class_id, i), img_as_ubyte(img))

            if "TrojAI" in clf:
                r = img[:, :, 0]
                g = img[:, :, 1]
                b = img[:, :, 2]
                img = np.stack((b, g, r), axis=2)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)

            img_list.append(img)

    inputs = torch.cuda.FloatTensor(np.concatenate(img_list, axis=0))
    pred = classifier(inputs)
    return jsonify({"prediction": pred.argmax(1).tolist()})
    ####################################################################################


@app.route('/crop2', methods=['POST'])
def generate_cropped_sample():
    data = request.get_json(force=True)
    # clf_id = int(data["clf_id"])
    clf = str(data["clf"])
    startX = int(data["startX"])
    startY = int(data["startY"])
    lenX = int(data["lenX"])
    lenY = int(data["lenY"])
    startX_v2 = int(data["startX_v2"])
    startY_v2 = int(data["startY_v2"])
    canvas_id_str = data["canvas_id_str"]

    startX_v2 = max(startX_v2 - int(lenX/2), 0)
    startY_v2 = max(startY_v2 - int(lenY/2), 0)

    img_i, img_j = int(canvas_id_str[-11])-1, int(canvas_id_str[-1])-1

    print("In function generate img with cropped patch")
    print("startX %d startY %d lenX %d lenY %d"%(startX, startY, lenX, lenY))

    adv_img = adv_img_list[img_i * 5 + img_j]
    cropped_patch = adv_img[startY:(startY+lenY),startX:(startX+lenX),:]

    ###############################################################################################
    # base = "/project_data/datasets/trojai/trojai-round0-dataset/id-%.8d/example_data"%clf_id
    if "TrojAI" in clf and "Poisoned" in clf:
        base = "models/trojai/poisoned/example_data"
    elif "TrojAI" in clf and "Clean" in clf:
        base = "models/trojai/clean/example_data"
    elif "ImageNet" in clf:
        base = "models/imagenet/example_data"

    out_img_list = []
    imgpath = os.path.join(base, "class_0_example_0.png")
    img = skimage.io.imread(imgpath)

    img = img - np.min(img)
    img = img / np.max(img)

    # img[100:(100+lenY),100:(100+lenX),:] = cropped_patch
    img[startY_v2:(startY_v2+lenY),startX_v2:(startX_v2+lenX),:] = cropped_patch

    new_img_pil = Image.fromarray((img*255.).astype(np.uint8))

    buffered = io.BytesIO()
    new_img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    out_img_list.append(img_str.decode("utf-8"))

    return jsonify({"out_img_list": out_img_list})

@app.route('/crop', methods=['POST'])
def generate_img_with_cropped_patch():
    data = request.get_json(force=True)
    # clf_id = int(data["clf_id"])
    clf = str(data["clf"])
    startX = int(data["startX"])
    startY = int(data["startY"])
    lenX = int(data["lenX"])
    lenY = int(data["lenY"])
    startX_v2 = int(data["startX_v2"])
    startY_v2 = int(data["startY_v2"])
    canvas_id_str = data["canvas_id_str"]

    startX_v2 = max(startX_v2 - int(lenX/2), 0)
    startY_v2 = max(startY_v2 - int(lenY/2), 0)

    img_i, img_j = int(canvas_id_str[-11])-1, int(canvas_id_str[-1])-1

    print("In function generate img with cropped patch")
    print("startX %d startY %d lenX %d lenY %d"%(startX, startY, lenX, lenY))

    adv_img = adv_img_list[img_i * 5 + img_j]
    cropped_patch = adv_img[startY:(startY+lenY),startX:(startX+lenX),:]

    ###############################################################################################
    # base = "/project_data/datasets/trojai/trojai-round0-dataset/id-%.8d/example_data"%clf_id
    if "TrojAI" in clf and "Poisoned" in clf:
        base = "models/trojai/poisoned/example_data"
    elif "TrojAI" in clf and "Clean" in clf:
        base = "models/trojai/clean/example_data"
    elif "ImageNet" in clf:
        base = "models/imagenet/example_data"

    out_img_list = []
    for class_id in range(5):
        for i in range(5):
            imgpath = os.path.join(base, "class_%d_example_%d.png"%(class_id, i))
            img = skimage.io.imread(imgpath)

            img = img - np.min(img)
            img = img / np.max(img)

            # img[100:(100+lenY),100:(100+lenX),:] = cropped_patch
            img[startY_v2:(startY_v2+lenY),startX_v2:(startX_v2+lenX),:] = cropped_patch

            new_img_pil = Image.fromarray((img*255.).astype(np.uint8))

            buffered = io.BytesIO()
            new_img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            out_img_list.append(img_str.decode("utf-8"))

    return jsonify({"out_img_list": out_img_list})

@app.route('/toclass', methods=['POST'])
def img_and_gradient():
    print("start")
    start = time.time()
    data = request.get_json(force=True)

    noise_sd = 1.00
    # clf_id = int(data['id'])
    clf = str(data["clf"])
    targeted = data['targeted']
    target = int(data['target'])
    epsilon = float(data['epsilon'])
    # steps = int(data['steps'])
    steps = 10
    # num_noise_vec = int(data['num_noise_vec'])
    num_noise_vec = 8 
    # tv = float(data["tv"])
    tv = 0
    startX = float(data["startX"])
    startY = float(data["startY"])
    lenX = float(data["lenX"])
    lenY = float(data["lenY"])

    step_size = 2 * epsilon / float(steps)

    print("startX %f startY %f lenX %f lenY %f"%(startX, startY, lenX, lenY))
    #############################################################################################
    # clf_path = "/project_data/datasets/trojai/trojai-round0-dataset/id-%.8d/model.pt"%clf_id
    if "TrojAI" in clf and "Poisoned" in clf:
        clf_path = "models/trojai/poisoned/model.pt"
    elif "TrojAI" in clf and "Clean" in clf:
        clf_path = "models/trojai/clean/model.pt"
    elif "ImageNet" in clf and "Poisoned" in clf:
        clf_path = "models/imagenet/poisoned/poisoned_model.pt"
    elif "ImageNet" in clf and "Clean" in clf:
        clf_path = "models/imagenet/clean/clean_model.pt" 
    ##################################################################
    # Load classification model
    global classifier

    if "TrojAI" in clf:
        classifier = torch.load(clf_path)

        checkpoint = torch.load(model_args_1['denoiser_path'])
        denoiser = get_architecture(checkpoint['arch'], 
                                    'imagenet', 
                                    depth=model_args_1['denoiser_depth'], 
                                    n_channels=model_args_1['denoiser_nc'])
        denoiser.load_state_dict(checkpoint['state_dict'])
    elif "ImageNet" in clf:
        model_ft = models.alexnet(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,5)
        normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        classifier = nn.Sequential(normalize, model_ft)
        checkpoint = torch.load(clf_path, map_location='cuda:0')
        classifier.load_state_dict(checkpoint["state_dict"])

        checkpoint = torch.load(model_args_1['denoiser_path'])
        denoiser = get_architecture(checkpoint['arch'], 
                                    'imagenet',
                                    depth=model_args_1['denoiser_depth'],
                                    n_channels=model_args_1['denoiser_nc'])
        denoiser.load_state_dict(checkpoint['state_dict'])

    model = torch.nn.Sequential(denoiser.module, classifier)

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print("model loaded")
    ##############################################################################################

    attacker = PGD_L2(steps=steps, max_norm=epsilon)

    # base = "/project_data/datasets/trojai/trojai-round0-dataset/id-%.8d/example_data"%clf_id
    if "TrojAI" in clf and "Poisoned" in clf:
        base = "models/trojai/poisoned/example_data"
    elif "TrojAI" in clf and "Clean" in clf:
        base = "models/trojai/clean/example_data"
    elif "ImageNet" in clf:
        base = "models/imagenet/example_data"

    out_img_list = []
    out_adv_class = []

    print("before loading images")

    try:
        img_list = []
        target_list = []

        for class_id in range(5):
            for i in range(5):
                imgpath = os.path.join(base, "class_%d_example_%d.png"%(class_id, i))
                img = skimage.io.imread(imgpath)
                if "TrojAI" in clf:
                    r = img[:, :, 0]
                    g = img[:, :, 1]
                    b = img[:, :, 2]
                    img = np.stack((b, g, r), axis=2)
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0)
                img = img - np.min(img)
                img = img / np.max(img)

                img_list.append(img)
                if targeted:
                    target_list.append(target)
                else:
                    target_list.append(class_id)

        inputs = torch.cuda.FloatTensor(np.concatenate(img_list, axis=0))
        targets = torch.cuda.LongTensor(np.array(target_list))
        batch = [inputs, targets]

        print("images loaded")

        ############################################################
        mask = torch.zeros_like(inputs[0:1,:,:,:]).cuda()

        l = inputs.shape[2]
        startX = int(startX * l)
        startY = int(startY * l)
        lenX = int(lenX * l)
        lenY = int(lenY * l)

        if lenX != 0 or lenY != 0:
            mask[:,:,startY:startY+lenY,startX:startX+lenX] = 1
            mask = (mask==0)
        else:
            mask = (mask==1)

        print("startX %d startY %d lenX %d lenY %d"%(startX, startY, lenX, lenY))
        ############################################################

        out_adv_class_smoothed, out_img_list, out_adv_img_list_np = generate_adv(model, classifier, batch, attacker, targeted, num_noise_vec, noise_sd, step_size, tv, mask)
        end=time.time()
        print("time spent is %f"%(end-start))

        global adv_img_list
        adv_img_list = out_adv_img_list_np

        return jsonify({'smoothed_adv_pred_class': out_adv_class_smoothed, 'adv_image_list': out_img_list})
    except:
        tb = traceback.format_exc()
        return tb

if __name__ == "__main__":
    pass
