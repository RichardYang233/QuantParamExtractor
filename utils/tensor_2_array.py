import torch

def get_tensor_size(tensor):

    dim = tensor.dim()
    if dim == 1:
        return (tensor.size(0), None, None, None)
    if dim == 2:
        return (tensor.size(0), tensor.size(1), None, None)
    if dim == 4:
        return (tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3))

def tensor_2_c_array(tensor: torch.tensor, fileName, arrayName, type: str):
    '''
    type 仅支持: int8_t, int32_t
    '''
    if type not in ['int8_t', 'int32_t']:
        raise ValueError(f"type '{type}' 不支持，仅支持 'int8_t' 或 'int32_t'")

    
    dim = get_tensor_size(tensor)

    tensor = tensor.view(-1)

    length = tensor.numel()

    with open(f"./parameter/{fileName}.h", "w") as f:
        # 头文件
        f.write(f"#ifndef {fileName.upper()}_H\n#define {fileName.upper()}_H\n\n")
        f.write(f"#include <stdint.h>\n\n")
        # 数组长度
        f.write(f"#define {arrayName.upper()}   {length}\n\n")
        # 尺寸说明
        f.write(f"// {dim[0]} * {dim[1]} * {dim[2]} * {dim[3]}\n")
        # 数组
        f.write(f"const {type} {arrayName}[{arrayName.upper()}] = {{\n")
        for i in range(length):
            f.write(f"{int(tensor[i])}, ")
            if (i + 1) % 50 == 0:
                f.write("\n")
        f.write("};\n\n#endif\n")

# test
if __name__ == "__main__":

    # data = torch.tensor([ [1, 2, 3],
    #                     [4, 5, 6] ])

    data = torch.tensor([[[[ 26, -19, -35, -15, -20],   # torch.Size([6, 1, 5, 5])
                        [ 46, -11, -36, -23, -41],
                        [ 16,  24, -19,   8,  38],
                        [ 57,  68,  61,  11,  30],
                        [ 16,  30,  66,  66,  55]]],

                        [[[-16, -19, -34,  -8, -24],
                        [ 26,  -4, -30, -24, -11],
                        [-10, -15, -35, -21,  16],
                        [ 50,  23,  28,  -9,   3],
                        [ 57,  59,  62,  63,  17]]],

                        [[[ 13,  18, -28,  15,  20],
                        [ 10, -28,   6,  -5,  17],
                        [ -9,  -9,   3, -18, -16],
                        [ 16, -20,  -8,  -5,   4],
                        [-20,   2,   8,  -9, -11]]],

                        [[[ 28,  17, -19, -18,   0],
                        [-13,  28,  33, -15,  30],
                        [ -7,  15, -23,   5,  21],
                        [  3, -16,  22,  -6, -22],
                        [  4, -29,   4, -22, -24]]],

                        [[[ 22, -11, -26, -22, -18],
                        [ 39,  31,  34,  13,  37],
                        [ 34,  20,   6,  30,  11],
                        [  3,  20,  16,  38,  -8],
                        [ 12, -29,  25,  -8,  -9]]],

                        [[[ 16,  82,  91,  18,  30],
                        [ 84, 108, 125,  66,  40],
                        [ 89, 127,  79,  79,   2],
                        [ 29,  52,  57,  62,   3],
                        [ -9,  29,  30,  -4,  13]]]], dtype=torch.int8)
    
    filename = "testdata"

    tensor_2_c_array(data, filename, "weight_int8", 'int8_t')

    data = data.view(-1)
    print(data)