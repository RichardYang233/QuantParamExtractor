import torch


def get_tensor_diff_count(tensor1, tensor2):
    '''
    获取两个 Tensor 中不相等元素的个数
    '''
    diff_count = torch.ne(tensor1, tensor2).sum()

    return diff_count.item()

def get_tensor_diff_index(tensor1, tensor2):
    '''
    获取两个 Tensor 中不相等元素的位置
    '''
    diff_mask = (tensor1 != tensor2)

    diff_index = torch.nonzero(diff_mask, as_tuple=False)

    return diff_index

