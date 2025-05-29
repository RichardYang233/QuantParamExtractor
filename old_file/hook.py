import torch
import copy
import functools

from models import *
from train import *
from eval import *
from dataset import *
from linear_quantize import *
from quantize_Layer import *


def add_range_recoder_hook(model):
    
    def _record_range(self, input, output, module_name):
        input = input[0]
        input_activation[module_name] = input.detach()
        output_activation[module_name] = output.detach()

    all_hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU)):
            all_hooks.append(layer.register_forward_hook(functools.partial(_record_range, module_name=name)))
    return all_hooks

def remove_hook(hooks):
    for hook in hooks:
        hook.remove()


def creat_hook(model, data_loader):
    hook_model = copy.deepcopy(model)
    hooks = add_range_recoder_hook(hook_model)  
    sample_data = iter(data_loader).__next__()[0]  # ([64, 1, 28, 28])
    # print(iter(data_loader).__next__()[1])
    with torch.no_grad():
        a = hook_model(sample_data)
        print(a.argmax(dim=1))
    return 




if __name__ == "__main__":

    # -------- load raw model -------- #

    # DataLoader

    train_loader, test_loader = get_DataLoader(BATCH)

    # Model with Params

    model = FCNet().to(device=DEVICE)
    checkpoint = torch.load('./parameter/FCNet_params.pt', weights_only=True)
    model.load_state_dict(checkpoint)
    model = copy.deepcopy(model)

    # -------- hook -------- #

    input_activation = {}
    output_activation = {}
    creat_hook(model, test_loader)

    quantized_backbone = []
    model_list = []
    model_name = []
    for name, layer in model.named_children():
        # print( name + ':', layer)
        model_list.append(layer)
        model_name.append(name)


    ptr = 0
    quantized_backbone.append(Reshape())
    for name, layer in model.named_children():

        if isinstance(layer, nn.Linear):
            # input 
            q_input, input_scale, input_zero_point = linear_quantization(input_activation[name], dtype=torch.int8)
            r_input = input_activation[name]
            # output
            if ptr < len(model_list) - 1 and isinstance(model_list[ptr + 1], nn.ReLU):
                q_output, output_scale, output_zero_point = linear_quantization(output_activation[model_name[ptr + 1]], dtype=torch.int8)
            else:
                q_output, output_scale, output_zero_point = linear_quantization(output_activation[name], dtype=torch.int8)
            
            # weight
            q_weight, weight_scale = linear_quantization_symmetric(layer.weight.data, dtype=torch.int8)
            r_weight = layer.weight.data

            # bias
            # bias_scale = input_scale * weight_scale
            # q_bias = linear_q_with_scale_and_zero_point(layer.bias.data, bias_scale, 0, dtype=torch.int32)
            q_bias, bias_scale = linear_quantization_symmetric(layer.bias.data, dtype=torch.int8)
            r_bias = layer.bias.data

            # build layer
            quantized_lin = QuantizedLinear(q_weight, q_bias,
                                            r_input, r_weight, r_bias,
                                            input_scale, input_zero_point,
                                            output_scale, output_zero_point)
            quantized_backbone.append(quantized_lin)

            ptr += 1
        elif isinstance(layer, nn.ReLU):
            quantized_backbone.append(Relu())
            ptr += 1
        else:
            raise NotImplementedError(type(layer))  
        
quant_model_creater = nn.Sequential(*quantized_backbone)

print(quant_model_creater)
int8_model_accuracy = evaluate(quant_model_creater, test_loader)
print(f"\nint8 model has accuracy={int8_model_accuracy:.2f}%\n")
    








    # print(input_activation["hidden_layer"])
    # print(input_activation["relu"] == output_activation["hidden_layer"])
    # print(input_activation["output_layer"] == output_activation["relu"])
    # print(output_activation["output_layer"])










        






        

    

