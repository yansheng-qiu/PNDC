#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models_pndc as models
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=100, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--et', default=20.0, type=float)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())





def findPNmap(prob, alpha_t, target):
    target = torch.argmax(target, dim=1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

    
    low_thresh = np.percentile(
    entropy[target != 0].cpu().numpy().flatten(), alpha_t)
    low_entropy_mask = (
        entropy.le(low_thresh).float())

    high_thresh = np.percentile(
        entropy[target != 0].cpu().numpy().flatten(), 100 - alpha_t,
    )
    high_entropy_mask = (
        entropy.ge(high_thresh).float())
    return low_entropy_mask.unsqueeze(1).cuda(), high_entropy_mask.unsqueeze(1).cuda()





    

def mask_4(Neg, Neg_fusion, mask):
    exit_modal_A,exit_modal_B,exit_modal_C,exit_modal_D = Neg

    pseudo_A = Neg_fusion - exit_modal_A
    pseudo_A_BCD = (pseudo_A > 0)

    pseudo_B = Neg_fusion - exit_modal_B
    pseudo_B_ACD = (pseudo_B > 0)


    pseudo_C = Neg_fusion - exit_modal_C
    pseudo_C_ABD = (pseudo_C > 0)


    pseudo_D = Neg_fusion - exit_modal_D
    pseudo_D_ABC = (pseudo_D > 0)

    pseudo_D_fusion = np.logical_and(pseudo_A_BCD.cpu(), np.logical_and(pseudo_B_ACD.cpu(), pseudo_C_ABD.cpu()))

    pseudo_C_fusion = np.logical_and(pseudo_A_BCD.cpu(), np.logical_and(pseudo_B_ACD.cpu(), pseudo_D_ABC.cpu()))

    pseudo_B_fusion = np.logical_and(pseudo_A_BCD.cpu(), np.logical_and(pseudo_C_ABD.cpu(), pseudo_D_ABC.cpu()))

    pseudo_A_fusion = np.logical_and(pseudo_B_ACD.cpu(), np.logical_and(pseudo_C_ABD.cpu(), pseudo_D_ABC.cpu()))

    pseudo_A = np.logical_or(pseudo_A_fusion.cpu(), exit_modal_A.cpu())

    pseudo_B = np.logical_or(pseudo_B_fusion.cpu(), exit_modal_B.cpu())

    pseudo_C = np.logical_or(pseudo_C_fusion.cpu(), exit_modal_C.cpu())

    pseudo_D = np.logical_or(pseudo_D_fusion.cpu(), exit_modal_D.cpu())


    return (torch.tensor(pseudo_A).cuda(), torch.tensor(pseudo_B).cuda(), torch.tensor(pseudo_C).cuda(), torch.tensor(pseudo_D).cuda())








def mask_3(Neg, Neg_fusion, mask):
    exit_modal = []
    for i in range(0,4):
        if mask[0,i]:
            exit_modal.append(Neg[i])
    exit_modal_A,exit_modal_B,exit_modal_C = exit_modal

    pseudo_A = Neg_fusion - exit_modal_A
    pseudo_A_BC = (pseudo_A > 0)

    pseudo_B = Neg_fusion - exit_modal_B
    pseudo_B_AC = (pseudo_B > 0)


    pseudo_C = Neg_fusion - exit_modal_C
    pseudo_C_AB = (pseudo_C > 0)



    pseudo_C_fusion = np.logical_and(pseudo_A_BC.cpu(), pseudo_B_AC.cpu()) 

    pseudo_B_fusion = np.logical_and(pseudo_A_BC.cpu(), pseudo_C_AB.cpu())

    pseudo_A_fusion = np.logical_and(pseudo_B_AC.cpu(), pseudo_C_AB.cpu())

    pseudo_A = np.logical_or(pseudo_A_fusion.cpu(), exit_modal_A.cpu())

    pseudo_B = np.logical_or(pseudo_B_fusion.cpu(), exit_modal_B.cpu())

    pseudo_C = np.logical_or(pseudo_C_fusion.cpu(), exit_modal_C.cpu())



    return (torch.tensor(pseudo_A).cuda(), torch.tensor(pseudo_B).cuda(), torch.tensor(pseudo_C).cuda())




def mask_2(Neg, Neg_fusion, mask):
    exit_modal = []
    for i in range(0,4):
        if mask[0,i]:
            exit_modal.append(Neg[i])
    exit_modal_A,exit_modal_B = exit_modal

    pseudo_A = Neg_fusion - exit_modal_A
    pseudo_A_B = (pseudo_A > 0)

    pseudo_B = Neg_fusion - exit_modal_B
    pseudo_B_A = (pseudo_B > 0)



    pseudo_B_fusion = pseudo_A_B 

    pseudo_A_fusion = pseudo_B_A

    pseudo_A = np.logical_or(pseudo_A_fusion.cpu(), exit_modal_A.cpu())

    pseudo_B = np.logical_or(pseudo_B_fusion.cpu(), exit_modal_B.cpu())

    return (torch.tensor(pseudo_A).cuda(), torch.tensor(pseudo_B).cuda())

def mask_1(Neg, Neg_fusion, mask):
    exit_modal = []
    for i in range(0,4):
        if mask[0,i]:
            exit_modal.append(Neg[i])
    exit_modal_A = exit_modal[0]

    pseudo_A_Ture = np.logical_or(Neg_fusion.int().cpu(), exit_modal_A.int().cpu())
    # pseudo_A_Ture = (pseudo_A.le(0))
    
    return torch.tensor(pseudo_A_Ture).cuda()


def mask_pos(Posmaps, Pos_fusion, mask):
    


    final_mask = Posmaps[0]

    all_mask = []


    for i in range(0,4):
        if mask[0,i]:
            curren_pos=Posmaps[i]

            mutal_pos = curren_pos - Pos_fusion

            curren_mask = (mutal_pos > 0)

            all_mask.append(curren_mask)




    for i, mask_curren in enumerate(all_mask):
        # print(i)
        if i == 0:
            fianl_mask = mask_curren
        else:
            final_mask = np.logical_and(final_mask.cpu(), mask_curren.cpu())


    
    return torch.tensor(fianl_mask).cuda()







def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2020', 'BRATS2018', 'BRATS2023']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = models.Model(num_cls=num_cls)
    # print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)


    if args.pretrain is not None:
        print('loading!!!!!!!!!!')
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])


    ##########Setting data
    if args.dataname in ['BRATS2020', 'BRATS2015', 'BRATS2023']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train2.txt'
        test_file = 'test2.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        test_score = AverageMeter()
        with torch.no_grad():
            logging.info('###########test set wi post process###########')
            for i, mask in enumerate(masks[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                dice_score = test_softmax(
                                test_loader,
                                model,
                                dataname = args.dataname,
                                feature_mask = mask,
                                mask_name = mask_name[::-1][i])
                test_score.update(dice_score)
            logging.info('Avg scores: {}'.format(test_score.avg))
            exit(0)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds, fuse_logits, seg_logits = model(x, mask)

        


            flair_pred, t1ce_pred, t1_pred, t2_pred = sep_preds

            flair_logits, t1ce_logits, t1_logits, t2_logits = seg_logits

            if torch.sum(mask[0].int())==4:
                alpha_flair = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t1ce = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t1 = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t2 = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_fusion = args.et * (
                    1 - epoch / args.num_epochs
                )                    


            elif torch.sum(mask[0].int())==3:
                alpha_flair = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t1ce = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t1 = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t2 = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_fusion = args.et * (
                    1 - epoch / args.num_epochs
                )                     

            elif torch.sum(mask[0].int())==2:
                alpha_flair = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t1ce = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t1 = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_t2 = args.et * (
                    1 - epoch / args.num_epochs
                )
                alpha_fusion =  args.et * (
                    1 - epoch / args.num_epochs
                )


            if torch.sum(torch.argmax(target.clone(), dim=1) != 0):
                with torch.no_grad():




                    Pos_flair, Neg_flair, = findPNmap(flair_pred, alpha_flair, target.clone())
                    Pos_t1ce, Neg_t1ce = findPNmap(t1ce_pred, alpha_t1ce, target.clone())
                    Pos_t1, Neg_t1 = findPNmap(t1_pred, alpha_t1, target.clone())
                    Pos_t2, Neg_t2 = findPNmap(t2_pred, alpha_t2, target.clone())
                    Pos_fusion, Neg_fusion = findPNmap(fuse_pred, alpha_fusion, target.clone())


                    Neg_fusion_pre = Neg_fusion * fuse_logits

                    # Pos_fusion_pre = Pos_fusion * fuse_logits

                    Neg = []
                    Neg.append(Neg_flair)
                    Neg.append(Neg_t1ce)
                    Neg.append(Neg_t1)
                    Neg.append(Neg_t2)

                    Pos = []

                    Pos.append(Pos_flair)
                    Pos.append(Pos_t1ce)
                    Pos.append(Pos_t1)
                    Pos.append(Pos_t2)

                    keys = ['flair_pred', 't1ce_pred', 't1_pred', 't2_pred']
                    Neg_result = {}

                    # print(Neg_fusion)

                    Pos_fusion_wrong = mask_pos(Pos, Pos_fusion, mask)
                    Pos_fusion_wrong = np.logical_or(Pos_fusion_wrong.cpu(), Neg_fusion.int().cpu())
                    Pos_fusion_wrong = Pos_fusion_wrong.cuda()

                    


                    # print(torch.sum(mask[0].int()))
                    if torch.sum(mask[0].int())==4:
                        alpha_flair = 30
                        alpha_t1ce = 25
                        alpha_t1 = 30
                        alpha_t2 =30
                        alpha_fusion = 15                    
                        Neg_all = mask_4(Neg, Neg_fusion, mask)
                        countone = 0
                        for index in range(0,4):
                            if mask[0,index]:
                                Neg_result[keys[index]] = Neg_all[countone]
                                countone += 1
                            else:
                                Neg_result[keys[index]] = 0

                    elif torch.sum(mask[0].int())==3:
                        alpha_flair = 30
                        alpha_t1ce = 25
                        alpha_t1 = 30
                        alpha_t2 =30
                        alpha_fusion = 18                     
                        Neg_all = mask_3(Neg, Neg_fusion, mask)
                        countone = 0
                        for index in range(0,4):
                            if mask[0,index]:
                                Neg_result[keys[index]] = Neg_all[countone]
                                countone += 1
                            else:
                                Neg_result[keys[index]] = 0
                    elif torch.sum(mask[0].int())==2:
                        alpha_flair = 30
                        alpha_t1ce = 25
                        alpha_t1 = 30
                        alpha_t2 =30
                        alpha_fusion =  23
                        Neg_all = mask_2(Neg, Neg_fusion, mask)
                        countone = 0
                        for index in range(0,4):
                            if mask[0,index]:
                                Neg_result[keys[index]] = Neg_all[countone]
                                countone += 1
                            else:
                                Neg_result[keys[index]] = 0
                if torch.sum(mask[0].int())==1:
                    sep_cross_loss2 = torch.zeros(1).cuda().float()
                else:
                    sep_cross_loss2 = torch.zeros(1).cuda().float()



                    
                    count = 0.0
                    if mask[0,0]:
                        count += 1.0
                        sep_cross_loss2 += criterions.softmax_weighted_loss(flair_pred*Neg_result['flair_pred'], target*Neg_result['flair_pred'], num_cls=num_cls)
                        
                    if mask[0,1]:
                        count += 1.0
                        sep_cross_loss2 += criterions.softmax_weighted_loss(flair_pred*Neg_result['t1ce_pred'], target*Neg_result['t1ce_pred'], num_cls=num_cls)
                        
                    if mask[0,2]:
                        count += 1.0
                        sep_cross_loss2 += criterions.softmax_weighted_loss(flair_pred*Neg_result['t1_pred'], target*Neg_result['t1_pred'], num_cls=num_cls)
                        
                    if mask[0,3]:
                        count += 1.0
                        sep_cross_loss2 += criterions.softmax_weighted_loss(flair_pred*Neg_result['t2_pred'], target*Neg_result['t2_pred'], num_cls=num_cls)
                        

                    sep_cross_loss2 = sep_cross_loss2/count

            else:
                sep_cross_loss2 = torch.zeros(1).cuda().float()


            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss



            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss




            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss


            loss = fuse_loss + sep_loss + prm_loss  + 0.4*sep_cross_loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)
        
        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))

if __name__ == '__main__':
    main()
