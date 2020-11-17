import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import wandb
from transformers import get_linear_schedule_with_warmup
from utils import *


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    getsch = get_cosine_schedule_with_warmup if args.scheduler =='cosine' else get_linear_schedule_with_warmup
    scheduler = getsch(optimizer, args.warmup, args.early_stop)

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.train()
            if not args.scatterlab:
                feature, target = batch.text, batch.label
                feature.t_(), target.sub_(1)  # batch first, index align
            else: # scatterlab data
                b, l, datasetids = batch
                feature, target = b.input_ids, l
                bsz = len(b.input_ids)

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/(bsz if args.scatterlab else batch.batch_size)
                wandb.log(
                    {
                        'train_step/lr': get_lr_from_optim(optimizer),
                        'train_step/acc':  accuracy,
                        'train_step/loss': loss.item()
                    }
                )
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             bsz if args.scatterlab else batch.batch_size))

            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                        save(model, args.save_dir, 'snapshot', steps)
                        return None
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        if not args.scatterlab:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
        else: # scatterlab data
            b, l, datasetids = batch
            feature, target = b.input_ids, l
            bsz = len(b.input_ids)

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter)*bsz if args.scatterlab else len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    wandb.log(
        {
            'dev_ckpt/acc':  accuracy,
            'dev_ckpt/loss': avg_loss
        }
    )
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    print(save_path)
    torch.save(model.state_dict(), save_path)
