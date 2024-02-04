from .models import ArcMarginModel
from .models import ResNet
from .models import IRBlock
from .models import SEBlock

# from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
# from .mobilefacenet import get_mbf
from .ir_se_net import resnet101

'''     
def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    else:
        raise ValueError()
'''

import importlib
import torch


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, torch.nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

'''
def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % (type(instance).__name__))

    return instance
'''