import torch

def get_tensor_size(tensor):

    dim = tensor.dim()
    if dim == 1:
        return tensor.size(0), None
    if dim == 2:
        return tensor.size(0), tensor.size(1)

def tensor_2_c_array(tensor: torch.tensor, fileName, arrayName, type: str):
    '''
    type 仅支持: int8_t, int32_t
    '''
    if type not in ['int8_t', 'int32_t']:
        raise ValueError(f"type '{type}' 不支持，仅支持 'int8_t' 或 'int32_t'")

    row, col = get_tensor_size(tensor)

    tensor = tensor.view(-1)

    length = tensor.numel()

    with open(f"./parameter/{fileName}.h", "w") as f:
        # 头文件
        f.write(f"#ifndef {fileName.upper()}_H\n#define {fileName.upper()}_H\n\n")
        f.write(f"#include <stdint.h>\n\n")
        # 数组长度
        f.write(f"#define {arrayName.upper()}   {length}\n\n")
        # 尺寸说明
        f.write(f"// {row} * {col}\n")
        # 数组
        f.write(f"const {type} {arrayName}[{arrayName.upper()}] = {{\n")
        for i in range(length):
            f.write(f"{int(tensor[i])}, ")
            if (i + 1) % 50 == 0:
                f.write("\n")
        f.write("};\n\n#endif\n")

# test
if __name__ == "__main__":

    data = torch.tensor([ [1, 2, 3],
                        [4, 5, 6] ])
    filename = "testdata"

    tensor_2_c_array(data, filename, "weight_int8")