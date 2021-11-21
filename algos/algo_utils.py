import torch


def sum_tensors(*tensors):
    # given torch tensors tensor_1, tensor_2, ...
    # calculate tensor_1 + tensor_2 + ...
    # the total number of tensors is variable
    return torch.sum(torch.stack(tensors, 0), 0)


def sum_lists(lists):
    # given a list of lists of tensors [list_1, list_2, ...]
    # where list_1 = [tensor_11, tensor_12, ...]
    #       list_2 = [tensor_21, tensor_22, ...]
    #       ...
    # return [tensor_11+tensor_21+ ...,
    #         tensor_12+tensor_22+ ...,
    #       ...]
    # the total number of lists is variable
    return list(map(sum_tensors, *lists))


def cat_tensors(tensor_1, tensor_2):
    return torch.cat((tensor_1, tensor_2), dim=1)


def cat_lists(list1, list2):
    # given two lists list1 = [tensor_11, tensor_12, ...]
    #                 list2 = [tensor_21, tensor_22, ...]
    # return [torch.cat((tensor_11, tensor_21)), torch.cat((tensor_12, tensor_22)), ...]
    return list(map(cat_tensors, *[list1, list2]))


def int1D_to_onehot(indices, depth):
    # given an integer-valued torch tensor, return the corresponding one-hot encoding
    one_hot = torch.zeros((torch.numel(indices), depth), dtype=torch.double)
    one_hot.scatter_(1, indices[:, None], 1)
    return one_hot


def int2D_to_grouponehot(indices, depths):
    # given a 2-D int tensor and a tuple of depths, first find the one-hot encoding for each dimension
    # then concatenate them (group one-hot)
    action_sample_onehot = int1D_to_onehot(indices=indices[:, 0], depth=depths[0])
    for ii in range(1, len(depths)):
        onehot = int1D_to_onehot(indices=indices[:, ii], depth=depths[ii])
        action_sample_onehot = torch.cat((action_sample_onehot, onehot), dim=1)
    return action_sample_onehot


def int3D_to_grouponehot(indices, depths):
    # analogous to int2D_to_grouponehot
    action_sample_onehot = int2D_to_grouponehot(indices=indices[:, 0, :], depths=depths)[:, None, :]
    for ii in range(1, indices.shape[1]):
        onehot = int2D_to_grouponehot(indices=indices[:, ii, :], depths=depths)[:, None, :]
        action_sample_onehot = torch.cat((action_sample_onehot, onehot), dim=1)
    return action_sample_onehot
