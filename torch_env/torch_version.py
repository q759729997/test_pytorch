import torch


if __name__ == "__main__":
    # 测试torch版本号
    print(torch.__version__)
    print(torch.cuda.is_available())
