import torch
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os
from CNN.create_cnn import model, normal_mean, normal_std
from create_work_directory import images_dir
import aiohttp
import aiofiles


class EarlyImplementationError(Exception):
    def __init__(self, tried_attribute="one", neccesary_atribute="other", cls="Class"):
        self.message = f"You've just tried to set '{tried_attribute}' before '{neccesary_atribute}' in {cls} object."

    def __str__(self):
        return self.message


class StyleTransfer:
    def __init__(self, model, normal_mean=(0.5, 0.5, 0.5), normal_std=(0.5, 0.5, 0.5),
                 content_weight=1, style_weight=1e5, device="cuda"):
        # We are not going to change the network weights,
        # so it's neccesary to turn off the gradient calculating
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model.to(device)
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.content_weight = content_weight  # alpha
        self.style_weight = style_weight  # beta
        self.device = device
        self.conv_layers = None
        self.content_image = None
        self.target_image = None
        self.style_image = None
        self.content_rep_layer = None
        self.style_weights = None

    def load_image(self, img_path, max_size=400, shape=None):
        """
        Load in and transform an image, making sure the image
        is <= max_size pixels in the x-y dims.
        """
        if "http" in img_path:
            response = requests.get(img_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)
        if shape is not None:
            size = shape

        transform = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.normal_mean, self.normal_std)])
        image = transform(image)[:3, :, :].unsqueeze(0)
        return image

    def im_convert(self, tensor):
        """ Display a tensor as an image. """
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array(self.normal_std) + np.array(self.normal_mean)
        image = image.clip(0, 1)
        return image

    def load_images(self, content_path, style_path, max_size=400):
        self.content_image = self.load_image(content_path, max_size=max_size).to(self.device)
        self.style_image = self.load_image(style_path, shape=self.content_image.shape[-2:]).to(self.device)

    def set_conv_layers(self, content_rep_layer, max_in_group=1, to_append=None, conv_layers=None):
        if conv_layers is None:
            conv_layers = {}
            n_group = 1
            n_in_group = 0
            for key, layer in self.model._modules.items():
                if "conv" in self.model._modules[key].__class__.__name__.lower() and n_in_group < max_in_group:
                    n_in_group += 1
                    conv_layers[key] = f"conv{n_group}_{n_in_group}"
                if "pool" in self.model._modules[key].__class__.__name__.lower():
                    n_group += 1
                    n_in_group = 0
            if to_append:
                for layer, alias in to_append.items():
                    conv_layers[layer] = alias

        if not content_rep_layer in conv_layers.values():
            raise ValueError(
                "The name of layer which will be use as a content reprezentaion should be in 'conv_layers' attribute" +
                "If neccesary, add it by 'to_append' method")

        self.conv_layers = conv_layers
        self.content_rep_layer = content_rep_layer

    def conv_layers_results(self, image, model, conv_layers=None):
        if conv_layers is None:
            conv_layers = self.conv_layers
        x = image
        conv_layers_results = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in conv_layers.keys():
                conv_layers_results[conv_layers[name]] = x
        return conv_layers_results

    def gram_matrix(self, conv_layer_result):
        # get the batch_size, depth, height, and width of the Tensor
        b, d, h, w = conv_layer_result.size()
        # reshape so we're multiplying the features for each channel
        conv_layer_result = conv_layer_result.view(b * d, h * w)
        # calculate the gram matrix
        gram = torch.mm(conv_layer_result, conv_layer_result.t())
        return gram

    def style_reprsentation(self, conv_results):
        style_rep = {key: self.gram_matrix(value).to(self.device) for key, value in conv_results.items()}
        return style_rep

    def set_style_weights(self, style_weights):
        if not hasattr(self, 'conv_layers'):
            raise EarlyImplementationError("style_weights", "conv_layers", "StyleTransfer")
        for layer_name in style_weights.keys():
            if not layer_name in self.conv_layers.values():
                raise ValueError(f"Property 'conv_layers' does not vontains convolutional layer {layer_name}")
        self.style_weights = style_weights

    def set_content_style_ratio(self, content_weight=1, style_weight=1e4):
        self.content_weight = content_weight  # alpha
        self.style_weight = style_weight  # beta

    def content_loss(self, content_representation, target_representation):
        return 1 / 2 * torch.mean((target_representation - content_representation) ** 2)

    def style_loss(self, style_grams, target_conv_results, style_weights=None):
        if style_weights is None:
            style_weights = self.style_weights
        style_loss = 0
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_conv_results[layer]
            target_gram = self.gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)
        return style_loss

    def transfer_style(self, target_image=None, n_steps=4000, show_every=None, optimizer=None):
        if target_image is None:
            self.target_image = self.content_image.clone().requires_grad_(True).to(self.device)
            if optimizer:
                raise ValueError("If you set an 'optimizer' parametr, it's neccesary to set 'target_image'")
        else:
            self.target_image = target_image
        if show_every is None:
            show_every = n_steps
        if optimizer is None:
            optimizer = optim.Adam([self.target_image], lr=0.007)
        content_representation = self.conv_layers_results(self.content_image, self.model, self.conv_layers)[
            self.content_rep_layer]
        style_grams = self.style_reprsentation(self.conv_layers_results(self.style_image, self.model, self.conv_layers))

        for step in range(1, n_steps + 1):
            target_conv_results = self.conv_layers_results(self.target_image, self.model, self.conv_layers)
            target_con_representation = target_conv_results[self.content_rep_layer]

            con_loss = self.content_loss(content_representation, target_con_representation)

            st_loss = self.style_loss(style_grams, target_conv_results, self.style_weights)

            tot_loss = self.content_weight * con_loss + self.style_weight * st_loss
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()


async def load_image(img_path, max_size=400, shape=None, normal_mean=normal_mean, normal_std=normal_std):
    if "http" in img_path:
        async with aiohttp.ClientSession() as session:
            async with session.get(img_path) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    image = Image.open(BytesIO(content)).convert('RGB')
    else:
        f = await aiofiles.open(img_path, mode='rb')
        image = await f.read()
        await f.close()
        image = Image.open(BytesIO(image)).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape:
        size = shape

    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(normal_mean, normal_std)])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


def im_convert(tensor, normal_mean=normal_mean, normal_std=normal_std):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array(normal_std) + np.array(normal_mean)
    image = image.clip(0, 1)

    return image


async def get_conv_layers(model, content_rep_layer, max_in_group=1, to_append=None, conv_layers=None):
    if conv_layers is None:
        conv_layers = {}
        n_group = 1
        n_in_group = 0
        for key, layer in model._modules.items():
            if "conv" in model._modules[key].__class__.__name__.lower() and n_in_group < max_in_group:
                n_in_group += 1
                conv_layers[key] = f"conv{n_group}_{n_in_group}"
            if "pool" in model._modules[key].__class__.__name__.lower():
                n_group += 1
                n_in_group = 0
        if to_append:
            for layer, alias in to_append.items():
                conv_layers[layer] = alias

    if not content_rep_layer in conv_layers.values():
        raise ValueError(
            "The name of layer which will be use as a content reprezentaion should be in 'conv_layers' attribute" +
            "If neccesary, add it by 'to_append' method")
    return conv_layers


async def conv_layers_results(image, model, conv_layers):
    x = image
    conv_layers_results = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in conv_layers.keys():
            conv_layers_results[conv_layers[name]] = x

    return conv_layers_results


async def gram_matrix(conv_layer_result):
    # get the batch_size, depth, height, and width of the Tensor
    b, d, h, w = conv_layer_result.size()
    # reshape so we're multiplying the features for each channel
    conv_layer_result = conv_layer_result.view(b * d, h * w)
    # calculate the gram matrix
    gram = torch.mm(conv_layer_result, conv_layer_result.t())

    return gram


async def style_reprsentation(conv_results, device):
    style_rep = {}
    for key, value in conv_results.items():
        style_rep[key] = await gram_matrix(value)
        style_rep[key] = style_rep[key].to(device)
    # style_rep = {key: await gram_matrix(value).to(device) for key, value in conv_results.items()}
    return style_rep


async def content_loss(content_representation, target_representation):
    return 1/2 * torch.mean((target_representation - content_representation)**2)


async def style_loss(style_grams, target_conv_results, style_weights):
    style_loss = 0
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_conv_results[layer]
        target_gram = await gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)
    return style_loss


async def transfer_style(content_image_path, style_image_path, model, conv_layers,
                         con_rep_layer,layers_weights, style_weight, content_weight=1,
                         target_image=None, n_steps=4000, show_every=None, optimizer=None, device='cuda'):
    # Loading images and switching their devices
    content_image = await load_image(content_image_path)
    content_image = content_image.to(device)
    style_image = await load_image(style_image_path, shape=content_image.shape[-2:])
    style_image = style_image.to(device)
    # Setting parameters with None value
    if target_image is None:
        target_image = content_image.clone().requires_grad_(True).to(device)
    if show_every is None:
        show_every = n_steps
    if optimizer is None:
        optimizer = optim.Adam([target_image], lr=0.003)

    # Creating the content representation of content image for comparing it with our synthesizing image
    content_conv_layers_results = await conv_layers_results(content_image, model, conv_layers)
    content_representation = content_conv_layers_results[con_rep_layer]

    # Creating the style representation of style image(as a group of gram matrices)
    # for comparing it with our synthesizing image
    style_conv_layers_results = await conv_layers_results(style_image, model, conv_layers)
    style_grams = await style_reprsentation(style_conv_layers_results, device)

    for step in range(1, n_steps + 1):
        # Creating the content and style representations of our synthesized image
        target_conv_results = await conv_layers_results(target_image, model, conv_layers)
        target_con_representation = target_conv_results[con_rep_layer]

        # Calculating content and style losses to tune our synthesized image
        con_loss = await content_loss(content_representation, target_con_representation)
        st_loss = await style_loss(style_grams, target_conv_results, layers_weights)

        # Changing our target image according to differences with content and style images
        tot_loss = content_weight * con_loss + style_weight * st_loss
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

    return target_image


async def syntez_image(user_data, user_id):
    # Defining the convolutional layers from which we will take
    # content and style representations
    conv_layers = await get_conv_layers(model=model, to_append={"21": "conv4_2"}, content_rep_layer="conv4_2")
    # Setting device for style transfer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Setting the weight of every conv. layer which can build style representation
    layers_weights = {'conv1_1': 1,
                      'conv2_1': 0.8,
                      'conv3_1': 0.4,
                      'conv4_1': 0.2,
                      'conv5_1': 0.2}
    params = {
        "content_image_path": user_data["content_image"],
        "style_image_path": user_data["style_image"],
        "model": model.to(device),
        "conv_layers": conv_layers,
        "con_rep_layer": "conv4_2",
        "layers_weights": layers_weights,
        "style_weight": user_data["style_weight"],
        "device": device,
        "n_steps": 4000
    }
    # Transfering style
    target_image = await transfer_style(**params)
    user_dir = f"{images_dir}\\user_{user_id}"
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)
    # Saving the result
    await save_image(im_convert(target_image), f"{user_dir}\\result_image.jpg")


async def load_vgg():
    vgg = models.vgg19(weights='IMAGENET1K_V1', progress=True)
    vgg = vgg.features
    return vgg


async def initialize_transfer(model="vgg", normal_mean=(0.485, 0.456, 0.406), normal_std=(0.229, 0.224, 0.225),
                              content_weight=1, style_weight=1e5):
    # Setting the device for style transfer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model == "vgg":
        model = await load_vgg()

    layers_weights = {'conv1_1': 1,
                      'conv2_1': 0.8,
                      'conv3_1': 0.4,
                      'conv4_1': 0.2,
                      'conv5_1': 0.2}

    transfer = StyleTransfer(model=model, normal_mean=normal_mean, normal_std=normal_std,
                             content_weight=content_weight, style_weight=style_weight,
                             device=device)
    transfer.set_conv_layers(max_in_group=1, to_append={"21": "conv4_2"}, content_rep_layer="conv4_2")
    transfer.style_weights = layers_weights

    return transfer


async def save_image(img, filename):
    plt.imsave(filename, img)

