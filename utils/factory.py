from methods.inflora import InfLoRA
from methods.rlora import RLoRA
from methods.dlora import DLoRA
from methods.vanilla import Vanilla
from methods.alora import ALoRA
from methods.llora import LLoRA
def get_model(model_name, args):
    name = model_name.lower()
    options = {
               'inflora': InfLoRA,
               'rlora': RLoRA,
               'dlora': DLoRA,
               'vanilla': Vanilla,
               'alora': ALoRA,
               'llora':LLoRA
               }
    return options[name](args)

