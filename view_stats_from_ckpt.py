import os
import errno
import os.path
import sys
import argparse
import torch

def get_stats(model_dict):
    overall_remaining, overall_nelement, stats = [], [], []
    print(f'remaining params per layers:')
    for k in model_dict.keys():
        if k[-4:] == 'mask':
            remaining = torch.sum(model_dict[k]).item()
            nelem = model_dict[k].nelement()
            stat = f'{k} {remaining} out of {nelem} -> {100. * remaining / nelem :.2f}%'
            stats.append(stat)
            overall_remaining.append(remaining)
            overall_nelement.append(nelem)
    if len(overall_remaining) != 0:
        starting_elem_num = sum(overall_nelement)
        remaining_par = sum(overall_remaining)
        masks_nelem = f'GLOBAL param num, as is summing up masks nelem {starting_elem_num}'
        final_remaining_par = f'GLOBAL remaining pars {remaining_par} -> {100. * remaining_par / starting_elem_num :.2f}%'
        stats.append(final_remaining_par)
        stats.append(masks_nelem)
    return stats

def main(args):
    ckpt_path = args.ckpt
    model_dict = torch.load(ckpt_path)
    stats = get_stats(model_dict)

    for s in stats:
        print(s)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--ckpt', type=str, help='checkpoint location. must be a pth file wich contains a state_dict')
  args = parser.parse_args()
  main(args)