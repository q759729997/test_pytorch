import torch
import torch.nn.functional as F

if __name__ == "__main__":
    pred = torch.FloatTensor([
        [-0.1603, -1.3247, 0.2010, 0.9240, -0.6396],
        [-0.7316, -1.6028, 0.2281, 0.3558, 1.2500],
        [-1.2943, -1.7350, -0.7085, 1.1269, 1.0782],
    ])
    print('max:{}'.format(pred.max(dim=1)))
    """
    torch.return_types.max(
    values=tensor([0.9240, 1.2500, 1.1269]),
    indices=tensor([3, 4, 3]))
    """
    arg_max_pred = torch.Tensor([3, 2, 3])
    target = torch.Tensor([3, 3, 3])
    print('pred size:{}'.format(pred.size()))  # torch.Size([3, 5])
    print('arg_max_pred size:{}'.format(
        arg_max_pred.size()))  # torch.Size([3])
    print('target size:{}'.format(target.size()))  # torch.Size([3])
    sfm = F.softmax(pred, dim=1)
    print('sfm size:{}'.format(sfm.size()))  # torch.Size([3, 5])
    print(sfm)
    """
    tensor([[0.1581, 0.0494, 0.2269, 0.4677, 0.0979],
        [0.0702, 0.0294, 0.1832, 0.2082, 0.5091],
        [0.0393, 0.0253, 0.0707, 0.4429, 0.4218]])
    """
    print('max:{}'.format(sfm.max(dim=1)))
    """
    torch.return_types.max(
    values=tensor([0.4677, 0.5091, 0.4429]),
    indices=tensor([3, 4, 3]))
    """
    logsfm = F.log_softmax(pred, dim=1)
    print('logsfm size:{}'.format(logsfm.size()))  # torch.Size([3, 5])
    print(logsfm)
    """
    tensor([[-1.8443, -3.0087, -1.4830, -0.7600, -2.3236],
        [-2.6568, -3.5280, -1.6971, -1.5694, -0.6752],
        [-3.2357, -3.6764, -2.6499, -0.8145, -0.8632]])
    """
    print('max:{}'.format(logsfm.max(dim=1)))
    """
    torch.return_types.max(
    values=tensor([-0.7600, -0.6752, -0.8145]),
    indices=tensor([3, 4, 3]))
    """
    predict_prob, predict = logsfm.max(dim=1)
    print('predict_prob exe:{}'.format(predict_prob.exp()))
    """
    tensor([0.4677, 0.5091, 0.4429])
    """
