import argparse
import os
import torch
import numpy as np
import wandb
import torch.nn as nn
import torch.optim as optim
import utils
import time

from datetime import datetime
from torchvision import models
from model import ConvNet, LeNet5
from tqdm import tqdm

def evaluate(model, iterator):
    model.eval()
    correct = 0
    total = 0
    times = []
    if device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for ix, data in enumerate(tqdm(iterator)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            if device == 'cuda':
                start.record()
                outputs = model(images)
                end.record()
                torch.cuda.synchronize()
                curr_time = start.elapsed_time(end)
            else:
                start = time.perf_counter()
                outputs = model(images)
                end = time.perf_counter()
                curr_time = end - start
            times.append(curr_time)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, times

def train(model, parameters_to_prune, iterator, optimizer, criterion, logging, workdir, epochs, test_iterator, val_iterator, step_num_for_epoch, initial_ckpt_acc, initial_ckpt_test_acc, num_all_params):

    acc_list, test_acc_list, ckpt_list = [initial_ckpt_acc], [initial_ckpt_test_acc], ['random_init']
    non_zero_list = [num_all_params]
    overall_steps_cntr = 0
    gamma = args.reg_gamma
    gamma_cntr = 0  # this counter goes on till a crop appens. it is needed to calculate the gamma decay

    init_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(iterator, 0):
            overall_steps_cntr += 1
            gamma_cntr += 1

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients and run model
            optimizer.zero_grad()
            outputs = model(inputs)

            # get CE loss
            loss_tmp = criterion(outputs, labels)
            if args.regularize:
                # get gradients
                loss_tmp.backward(retain_graph=True)
                # decay gamma over the epoch
                if args.gamma_decay:
                    gamma = utils.gamma_decay(args.reg_gamma, gamma_cntr, step_num_for_epoch)
                # add regularizer term using precomputed gradients
                reg = utils.get_regularizer(model, gamma=gamma, alpha=args.exp_alpha, step=overall_steps_cntr)

                # get the regularization loss
                loss = loss_tmp + reg

                # clean the gradients since now we want to have the gradient for the regularized loss
                optimizer.zero_grad()
            else:
                loss = loss_tmp

            # get the loss gradients and optimize
            loss.backward()
            optimizer.step()

            # logging statistics
            running_loss += loss.item()

            if i % args.log_interval == args.log_interval -1:
                # logging every log_interval
                logging(f'[{epoch + 1}, {i + 1}, {overall_steps_cntr}] loss (with reg): {running_loss / args.log_interval :.4f} gamma {gamma :.6f}')
                running_loss = 0.0
                wandb.log({"training_loss": running_loss / args.log_interval, "training_gamma": gamma}, step=overall_steps_cntr)

            if (i % args.eval_interval == args.eval_interval -1) and val_iterator:
                # evaluate both in val and in test
                eval_acc, eval_times = evaluate(model, val_iterator)
                test_acc, test_times = evaluate(model, test_iterator)

                model.train()
                elapsed_time = time.time() - init_time

                acc_list.append(eval_acc)
                test_acc_list.append(test_acc)

                # log stuff
                eval_string = f' [{epoch + 1}, {i + 1}, {overall_steps_cntr}] validation accuracy: {eval_acc}%, test accuray: {test_acc}% time from script beginning {elapsed_time:.4f}s. eval_times:{np.mean(eval_times):.6f}, std:{np.std(eval_times):.6f}, test_times:{np.mean(test_times):.6f}, std:{np.std(test_times):.6f}'
                logging(eval_string)
                wandb.log({"eval_acc": eval_acc}, step=overall_steps_cntr)

                # checkpoint log
                ckpt_name = f'{args.dataset}_{args.model}_{overall_steps_cntr}_val{eval_acc:.2f}_test{test_acc:.2f}.pth' if args.regularize else f'{args.dataset}_{args.model}_{overall_steps_cntr}_{eval_acc:.2f}_test{test_acc:.2f}_noreg.pth'
                ckpt_list.append(ckpt_name)
                PATH = os.path.join(workdir, ckpt_name)
                torch.save(model.state_dict(), PATH)

                deleted_ckpt = utils.check_max_n_ckpt(workdir, args.patience)
                logging(f'deleted the following ckpt: {deleted_ckpt}')
                # if latest patience acc are decreasing stop
                if acc_list[-args.patience:] == sorted(acc_list[-args.patience:], reverse=True) and (len(acc_list) >= args.patience):
                    logging(f'ho finito la pazienza {acc_list}')
                    break

                if args.regularize:
                    if eval_acc > args.acc_lower_bound:
                        logging(f'model\'s acc {eval_acc}% is greater then lower bound:{args.acc_lower_bound}%')

                        # the threshold is intended as a percentage
                        param_under_threshold = args.threshold

                        if param_under_threshold > 0:
                            # save the model before the crop, because after it the acc will surely drop. so you'd have to recompute it.
                            # this way instead we have the acc and the num of params of a ckpt before cropping it. at the
                            # training end we will save the model so to get also the last ckpt even if no cropping is happening.
                            torch.save(model.state_dict(), PATH)
                            # get non zero using masks
                            non_zero_params = utils.sum_masks(parameters_to_prune)

                            # when cropping log accuracy on val and test and non_zero_params
                            non_zero_list.append(non_zero_params)

                            wandb.log({"remaining_params": non_zero_params * 100 / num_trinable_params}, step=overall_steps_cntr)
                            logging(f'\n before cropping {param_under_threshold} params we have {non_zero_params} remaining params -> {non_zero_params * 100 / num_trinable_params :.4f}% and eval_acc: {eval_acc}, test_acc: {test_acc}\n')
                            logging(f'PROCEEDING with CROPPING {param_under_threshold} params.')
                            utils.crop(parameters_to_prune, param_under_threshold)

                            # reset gamma decay
                            gamma = args.reg_gamma
                            gamma_cntr = 0
                        else:
                            logging(f' NOT CROPPING: the param_under_threshold {param_under_threshold} value must be positive to crop')
                            non_zero_list.append(non_zero_list[-1])
                    else:
                        logging(f'model\'s val acc {eval_acc}% is NOT greater than lower bound {args.acc_lower_bound}%')
                        non_zero_list.append(non_zero_list[-1])
                else:
                    non_zero_list.append(non_zero_list[-1])

    logging(f'final stats at training end:')
    stats = utils.get_stats(parameters_to_prune)
    for s in stats:
        logging(s)
    non_zero_params = utils.sum_masks(parameters_to_prune)

    # the final checkpoint is treated as a normal, not pruned ckpt
    utils.fix_pruning(parameters_to_prune)
    eval_acc, eval_times = evaluate(model, val_iterator)
    test_acc, test_times = evaluate(model, test_iterator)

    non_zero_list.append(non_zero_params)
    acc_list.append(eval_acc)
    test_acc_list.append(test_acc)

    ckpt_name = f'{args.dataset}_{args.model}_val{eval_acc:.2f}_test{test_acc:.2f}_end_training.pth' if args.regularize else f'{args.dataset}_{args.model}_val{eval_acc:.2f}_test{test_acc:.2f}_end_training_noreg.pth. eval_times:{np.mean(eval_times):.6f}, std:{np.std(eval_times):.6f}, test_times:{np.mean(test_times):.6f}, std:{np.std(test_times):.6f}'
    ckpt_list.append(ckpt_name)
    PATH = os.path.join(workdir, ckpt_name)
    torch.save(model.state_dict(), PATH)

    return acc_list, test_acc_list, non_zero_list, ckpt_list

def main(args):
    torch.manual_seed(args.seed)
    global device, num_trinable_params

    wandb.init(project="esann_experiment", entity="giobin")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # get work_dir and add time to dirname
    workdir = f'{args.work_dir}/{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    # get logger
    logging = utils.get_logger(log_path=os.path.join(workdir, 'log.txt'))
    logging('device: {}'.format(device))

    param_log_string = ''
    logging('===================')
    for ar in vars(args):
        logging(f'--{ar}, {getattr(args, ar)}')
        param_log_string += f'--{ar}, {getattr(args, ar)} \n'
    logging('===================')

    # get data
    trainset, testset, validationset, trainloader, testloader, validationloader = utils.get_data(args.dataset, args.batch_size, train_path=args.train_path, test_path=args.test_path)
    step_num_for_epoch = len(trainset) / args.batch_size

    if args.model == 'conv':
        model = ConvNet().to(device)
    elif args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
        if args.no_dropout:
            classifier_no_drop = utils.sequential_no_dropout(model.classifier)
            model.classifier = classifier_no_drop
        model.to(device)
    else:
        # default to leNet5
        model = LeNet5().to(device)

    parameters_to_prune = utils.select_pruning_params(model, model_type=args.model)
    logging(f'there are {len(parameters_to_prune)} parameters_to_prune: {parameters_to_prune}')

    if args.ckpt:
        logging(f'loading model from {args.ckpt}')
        if args.model == 'vgg16':
            logging(f'since vgg16 model, we check if the ckpt already contains masks and apply them beforehand.')
            ckpt = torch.load(args.ckpt)
            utils.init_vgg_with_pruned_buffer(model, ckpt)
        elif args.model == 'lenet5':
            ckpt_state_dict = torch.load(args.ckpt)
            logging('preparing lenet5 to have all the _mask and _orig from pruned ckpt')
            model = utils.load_pruned_ckpt_and_inject_in_new_model(ckpt_state_dict, model, 'lenet5')
            model(torch.rand(100, 1, 28, 28).to(device)) # need to be done in order to force model weights to the actual _mask * _orig tensor (needed just this time)
        else:
            raise Exception('the heck are you trying to load?')

    logging('--'*20 + 'model info' + '--'*20)
    logging(f'model {model}')
    logging('--' * 20 + 'train set info' + '--' * 20)
    logging(f'trainset {trainset}')
    logging('--' * 30)
    logging('--' * 20 + 'valid set info' + '--' * 20)
    logging(f'validationset {validationset}')
    logging('--' * 30)
    logging('--' * 20 + 'test set info' + '--' * 20)
    logging(f'testset {testset}')
    logging('--' * 30)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    num_all_params = utils.count_parameters(model, trainable=False)
    num_trinable_params = utils.count_parameters(model)

    logging(f'num of model parameters: {num_all_params} (also 0 params are counted here)')
    logging(f'num of trainable model parameters: {num_trinable_params} (also 0 params are counted here)')

    eval_acc, eval_times = evaluate(model, validationloader)
    test_acc, test_times = evaluate(model, testloader)
    logging(f'eval_acc before training {eval_acc}, and test acc {test_acc}. eval_times:{np.mean(eval_times):.6f}, std:{np.std(eval_times):.6f}, test_times:{np.mean(test_times):.6f}, std:{np.std(test_times):.6f}')
    model.train()

    remining = utils.sum_masks(parameters_to_prune)
    logging(f'remaining params {remining}')

    # log gradients and weights
    #wandb.watch(models=model, criterion=criterion, log="all", log_freq=100)

    acc_list, test_acc_list, non_zero_list, ckpt_list = train(model=model,
                                    parameters_to_prune=parameters_to_prune,
                                    iterator=trainloader,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    logging=logging,
                                    workdir=workdir,
                                    epochs=args.epochs,
                                    test_iterator=testloader,
                                    val_iterator=validationloader,
                                    step_num_for_epoch=step_num_for_epoch,
                                    initial_ckpt_acc=eval_acc,
                                    initial_ckpt_test_acc=test_acc,
                                    num_all_params=remining)

    logging(f'accuracy validation: {acc_list}, \ntest: {test_acc_list}')
    logging(f'remaining params: {non_zero_list} \n last number of each list is at the absolute training end.')


    # test acc when best val ckpt is used
    assert len(test_acc_list) == len(acc_list) == len(ckpt_list) == len(non_zero_list)
    best_val_idx = np.argmax(acc_list)
    best_test_acc = test_acc_list[best_val_idx]
    best_ckpt_name = ckpt_list[best_val_idx]
    param_num = non_zero_list[best_val_idx]
    logging(f'best val acc is: {np.max(acc_list)}, the corresponding test acc is {best_test_acc}, percentage of remaining params: {100 * param_num / num_trinable_params}%, -> {num_trinable_params / param_num}x')
    logging(f'best ckpt name: {best_ckpt_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', default='lenet5', choices=['lenet5', 'conv', 'vgg16'])
    parser.add_argument('-e','--epochs', type=int, default=30, help="number of epochs to train")
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'imagenet', 'fashion-mnist'])
    parser.add_argument('-tr','--train_path', type=str)
    parser.add_argument('-te','--test_path', type=str)
    parser.add_argument('--ckpt', default=None, help="training ckpt to start from")
    parser.add_argument('--acc_lower_bound', type=float, default=70.00, help="lower acceptable acc for cropping")
    parser.add_argument('--reg_gamma', type=float, default=0.005, help="regularizer term weight")
    parser.add_argument('--gamma_decay', action='store_true', help="if you want gamma decay or not")
    parser.add_argument('--exp_alpha', type=float, default=1., help="wheight to multiply exp argument. if regularization type hyperbole is the scalar that multiply the derivative")
    parser.add_argument('-t','--threshold', type=float, default=0.001, help="threshold to use to crop weights")
    parser.add_argument('-r','--regularize', action='store_true')
    parser.add_argument('-nod','--no_dropout', action='store_true', help='get rid of dropout layers')
    parser.add_argument('-b','--batch_size', type=int, default=4, help="batch dimension")
    parser.add_argument('-s','--seed', type=int, default=1234, help="seed")
    parser.add_argument('-o','--optim', default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('-mo', '--momentum', type=float, default=0.9, help="optim momentum")
    parser.add_argument('--log_interval', type=int, default=100, help="log interval in number of batches")
    parser.add_argument('--eval_interval', type=int, default=250, help="eval interval in number of batches")
    parser.add_argument('--work_dir', type=str, default='workdir', help="dir where to store model ckpt")
    parser.add_argument('-p','--patience', type=int, default=25, help="patience")
    args = parser.parse_args()
    main(args)