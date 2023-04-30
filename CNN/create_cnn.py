from torchvision import models


def load_vgg():
    """
    Function that loads pretrained vgg19 model
    :return:
    tuple[torch.nn.Sequential, tuple[float, float, float], tuple[float, float, float]]
    """
    vgg = models.vgg19(weights='IMAGENET1K_V1', progress=True)
    vgg = vgg.features
    # Parameters which were used for normalizing images before training vgg19 model.
    normal_mean = (0.485, 0.456, 0.406)
    normal_std = (0.229, 0.224, 0.225)
    return vgg, normal_mean, normal_std


def load_castom_model():
    """
    Function that loads pretrained self-made model
    :return:
    tuple[torch.nn.Sequential, tuple[float, float, float], tuple[float, float, float]]
    """
    castom_model = None
    # Parameters which were used for normalizing images before training castom model.
    normal_mean = ()
    normal_std = ()
    return castom_model, normal_mean, normal_std


def load_model(model="vgg"):
    """
    The function that loads one of possible models and
    returns it with params of images normalizing
    which were used while model's training.
    :param model: one of two variants ('vgg' or 'castom') where
               "vgg" - VGG19 from torchvision framework
               "castom" - self-made or loaded from other sources cnn.
    :return: tuple[torch.nn.Sequential, tuple[float, float, float], tuple[float, float, float]]
    """
    if model == "vgg":
        model, normal_mean, normal_std = load_vgg()
    elif model == "castom":
        model, normal_mean, normal_std = load_castom_model()
    else:
        raise ValueError("It's necessary to set 'model' parameter to 'vgg' or 'castom'")
    for param in model.parameters():
        param.requires_grad_(False)
    return model, normal_mean, normal_std


model, normal_mean, normal_std = load_model()
