import argparse
import sys
import time
import numpy as np
import torch

from tqdm import tqdm

from torchvision import models

def logging(str):
    pass


def get_input(model_name, b_s):
    if model_name == 'resnet50':
        return torch.rand(b_s, 3, 256, 256)
    else:
        return torch.rand(b_s, 1, 28, 28)


def test_inference(model_name, ckpt_path, device, tries=100, b_s=100):
    if device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    print(f"preparing model")
    model = models.resnet50(pretrained=True)
    if ckpt_path is not None:
        print(f"using ckpt {ckpt_path}")
        ckpt_state_dict = torch.load(ckpt_path)
        model.load_state_dict(ckpt_state_dict)
    print(f"model ready")

    inp = get_input(model_name, b_s)
    inp = inp.to(device)
    model = model.to(device)
    model.eval()
    _ = model(inp)

    if device == 'cuda':
        start.record()
        for i in tqdm(range(tries)):
            _ = model(inp)
        end.record()
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)
        #times[0, i] = curr_time
    else:
        start = time.perf_counter()
        for i in tqdm(range(tries)):
            _ = model(inp)
        end = time.perf_counter()
        curr_time = end - start
        #times[0, i] = curr_time
    return curr_time/tries, 0


def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)

    print(f'{args.model} - {args.device}')
    print(f'{args.ckpt}')
    medium, std = test_inference(args.model, args.ckpt, device, tries=args.tries, b_s=args.batch_size)
    print(f'medium:{medium:.8f}; std:{std:.8f}')
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('-tries', type=int, default=1000, help="number of inferences")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="batch dimension")
    parser.add_argument('-m', '--model', default='resnet50', choices=['resnet50'])
    parser.add_argument('--ckpt', default=None, help="ckpt")
    args = parser.parse_args()
    main(args)
