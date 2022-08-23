import matplotlib.pyplot as plt

import functools
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.utils.prune as prune
import os
import wandb
from torch import nn
from glob import glob

def save_hist(data, path, title, color='#0504aa'):
    n, bins, patches = plt.hist(x=data, bins=200, color=color)
    plt.grid(axis='y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)

    path_to_hist = os.path.join(path, title)
    plt.savefig(path_to_hist)
    plt.clf()
    return


transform_images_mnist = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

transform_images_imagenet_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

transform_images_imagenet_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

def get_data(dataset, batch_size, train_path=None, test_path=None):
    if dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                     download=True, transform=transform_images_mnist)
        validationset, trainset = torch.utils.data.random_split(trainset, [5000, 55000])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, pin_memory=True, num_workers=8)
        validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                                  shuffle=False, pin_memory=True, num_workers=8)

        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                             download=True, transform=transform_images_mnist)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform_images_mnist)
        # validationset, trainset = torch.utils.data.random_split(trainset, [5000, 55000])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, pin_memory=True, num_workers=8)

        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=transform_images_mnist)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
        validationset = testset
        validationloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, pin_memory=True, num_workers=8)

    elif dataset == 'imagenet':
        assert train_path
        assert test_path
        trainset = torchvision.datasets.ImageFolder(
            train_path,
            transform_images_imagenet_train)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, pin_memory=True, num_workers=8)

        testset = torchvision.datasets.ImageFolder(
            test_path,
            transform_images_imagenet_valid)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)

        validationset = torch.utils.data.Subset(testset, indices=range(0, 5000))
        validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    else:
        raise Exception(f'dataset {dataset} not supported yet')

    return trainset, testset, validationset, trainloader, testloader, validationloader


def count_parameters(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def count_params_under_threshold(parameters_to_prune, threshold):
    count = 0
    param_modules = list(set([p[0] for p in parameters_to_prune]))
    for m in param_modules:
        pruned = prune.is_pruned(m)
        t = m.named_parameters()
        for name, tnsr in t:
            if pruned:
                assert name[-4:] == 'orig'
                count += tnsr[torch.abs(tnsr) < threshold].nelement()
            else:
                count += tnsr[torch.abs(tnsr) < threshold].nelement()
    return count

def gamma_decay(initial_gamma, current_step, step_num_for_epoch):
    gamma = initial_gamma - (initial_gamma/step_num_for_epoch)*current_step
    if gamma < 0.:
        gamma = 0.
    return gamma

def get_regularizer(model, gamma=0.5, alpha=0.1, step=1):
    regularizer = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            # get the gradient
            d_p = torch.abs(param.grad.data)
            # get the param square elem-wise
            p_square = param**2
            # gamma* 1/1+(-dL/dw) * p_square
            sparsity_importance = 1. / (1. + alpha*(-d_p))
            regularizer += gamma * torch.sum(sparsity_importance * p_square)

    return regularizer

def crop(parameters_to_prune, amount):
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return

def fix_pruning(parameters_to_fix):
    for p in parameters_to_fix:
        if prune.is_pruned(p[0]):
            prune.remove(*p)
    return

def add_param_to_pruning(module, weight=True, bias=True):
    parameters_to_prune = []
    if weight and hasattr(module, 'weight'):
        if module.weight is not None:
            parameters_to_prune.append((module, 'weight'))
    if bias and hasattr(module, 'bias'):
        if module.bias is not None:
            parameters_to_prune.append((module, 'bias'))
    return parameters_to_prune

def select_pruning_params(model, model_type):
    parameters_to_prune = []
    if model_type == 'lenet5':
        for name, module in model.named_modules():
            if hasattr(model, name):
                m = getattr(model, name)
                parameters_to_prune += add_param_to_pruning(m, weight=True, bias=True)
    elif model_type == 'vgg16':
        sequential = model.features
        classifier = model.classifier
        for mod in [sequential, classifier]:
            for name, module in mod.named_modules():
                if hasattr(mod, name):
                    m = getattr(mod, name)
                    parameters_to_prune += add_param_to_pruning(m, weight=True, bias=True)
    else:
        raise Exception(f'{model_type} not yet implemented')
    return parameters_to_prune

def init_vgg_with_pruned_buffer(original_model, ckpt):
    ckpt_keys = list(ckpt)
    sequential = original_model.features
    classifier = original_model.classifier
    for mod, mod_name in zip([list(sequential.named_modules())[1:], list(classifier.named_modules())], ['features', 'classifier']):
        for name, module in mod:  # (('0', Linear(...)),('1', Dropout(...)),..)
            # set weight and bias to be equal to weight_orig and bias_orig from ckpt
            weight_orig_name = f'{mod_name}.{name}.weight_orig'
            bias_orig_name = f'{mod_name}.{name}.bias_orig'
            if weight_orig_name in ckpt_keys:
                module.weight.data = ckpt[weight_orig_name]
                module.bias.data = ckpt[bias_orig_name]

                weight_mask_name = f'{mod_name}.{name}.weight_mask'
                weight_mask = ckpt[weight_mask_name]
                bias_mask_name = f'{mod_name}.{name}.bias_mask'
                bias_mask = ckpt[bias_mask_name]
                m = prune.custom_from_mask(module, name='weight', mask=weight_mask)
                b = prune.custom_from_mask(m, name='bias', mask=bias_mask)
                module = b
    return

def sum_masks(parameters_to_prune):
    non_zero_params = 0
    param_modules = list(set([p[0] for p in parameters_to_prune]))  #set because modules are repeated 2 times
    for m in param_modules:
        pruned = prune.is_pruned(m)
        if pruned:
            t = m.named_buffers()
            for name, tnsr in t:
                assert name[-4:] == 'mask'
                non_zero_params += torch.sum(tnsr).item()
        else:
            t = m.named_parameters()
            for name, tnsr in t:
                non_zero_params += tnsr.nelement()
    return non_zero_params

def get_stats(parameters_to_prune):
    overall_remaining, overall_nelement, stats = [], [], []
    param_modules = list(set([p[0] for p in parameters_to_prune]))
    for m in param_modules:
        pruned = prune.is_pruned(m)
        if pruned:
            t = m.named_buffers()
            for name, tnsr in t:
                assert name[-4:] == 'mask'
                remaining = torch.sum(tnsr).item()
                nelem = tnsr.nelement()
                stat = f'{m} {remaining} out of {nelem} -> {100. * remaining / nelem :.2f}%'
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

def load_pruned_ckpt_and_inject_in_new_model(ckpt_state_dict, model, model_type):
    """
    given a ckpt_state_dict which contains weigths, _orig and _mask, load them with all the whistles in the given model_params
    :param ckpt_state_dict: state dict of the checkpoint. (could come from a pruned model)
    :param model: the standard model_params (from original, never pruned model) we need to change with the ckpt_state_dict.
    :param model_type: is it vgg16, convnet, lenet5...
    :return: the loaded model. if ckpt_state_dict is coming from a pruned model, the returned model has the _orig and _mask tensors.
    """
    # check if the state_dict contains pruned params or is just a normal ckpt
    pruned = False
    for k in ckpt_state_dict.keys():
        if '_orig' in k:
            pruned = True
            break

    if pruned:
        # get model params
        model_params = select_pruning_params(model, model_type)

        # make params like if they have been pruned before
        for p in model_params:
            prune.identity(p[0], p[1])

    # load state dict into model
    model.load_state_dict(ckpt_state_dict)
    return model


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def sequential_no_dropout(module):
    layers = []
    for l in module:
        if not isinstance(l, nn.Dropout):
            layers.append(l)
    classifier_no_drop = nn.Sequential(*layers)
    return classifier_no_drop

def check_max_n_ckpt(path, patience):
    ckpts = glob(os.path.join(path, '*.pth'))
    latest_files = []
    if len(ckpts) > patience + 1:
        latest_files = sorted(ckpts, key=os.path.getctime)
        for f in latest_files[:-patience - 1]:
            os.remove(f)
    return latest_files[: -patience -1]
