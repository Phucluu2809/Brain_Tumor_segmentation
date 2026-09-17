import argparse
import wandb
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from inspect import currentframe, getframeinfo
import cv2
import matplotlib.pyplot as plt
wandb.init(project="TransUnet")
 
def trainer_synapse(args, model, snapshot_path):
    # print("importing synapse")
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    # print("loaded")
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
 
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        # print("epoch: ",epoch_num)
        for i_batch, sampled_batch in enumerate(trainloader):
            # print("batch # ", i_batch)
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            wandb.log({"iteration": iter_num})
            wandb.log({"lr" : lr_ })
            wandb.log({"total_loss" : loss})
            wandb.log({"loss_ce" : loss_ce,})

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 31 == 0:
                image = image_batch[1, 0:1, :, :]
              
                # print(image.size())
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
            
                img=image.cpu().numpy()
                img=np.transpose(img,(2,1,0))
                img = (img * 255).astype('uint8')
                # print(img)
                
                cv2.imwrite("img.png", img)
                
                train_img=wandb.Image(image)
                # print(train_img.shape)

                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                writer.add_image('train/Prediction_2', outputs[1, ...], iter_num)
                pred_img=(outputs[1, ...])
                # print(pred_img.sum())
                pred_img = pred_img.cpu().numpy()
                
                # pred_img=np.asarray(pred_img)
                pred_img = np.transpose(pred_img, (2,1, 0))
                # pred_img=(pred_img/8)
                # pred_img = (pred_img * 255).astype('uint8')
            
                # print(type(pred_img))
                # print(pred_img.shape)
                print("pred_sum", pred_img.sum())
                alpha=20
                color_map = cv2.COLORMAP_BONE  # You can change this to any of the available color maps


                pred_img = cv2.convertScaleAbs(pred_img, alpha=alpha, beta=0)
                pred_img = cv2.applyColorMap(pred_img, color_map)

                cv2.imwrite('pred.png', pred_img)
                # plt.imsave("pred_test.png", pred_img, cmap='gray')  # Assuming a grayscale image
                
                pred_img=wandb.Image(pred_img)
               
                # wandb.log({"prediction": pred_img})
                
                
                # print(pred_img.shape)


                labs = label_batch[1, ...].unsqueeze(0)                 
                writer.add_image('train/GroundTruth', labs, iter_num)
                print("gtsum",labs.sum())
                gt_img = labs.cpu().numpy()
                gt_img = np.transpose(gt_img, (2, 1, 0))
                # plt.imsave("gt_test.png", gt_img, cmap='gray')

                # print(gt_img) 
                # gt_img=cv2.add(gt_img,brightness)
                # cv2.imwrite('../ground_truths/gt_{}.png'.format(i_batch), gt_img)
                
                gt_img = cv2.convertScaleAbs(gt_img, alpha=alpha, beta=0)
                gt_img = cv2.applyColorMap(gt_img, color_map)



                cv2.imwrite('gt.png', gt_img)

                # gt_img=wandb.Image(gt_img)
                
                # wandb.log({"ground truth": gt_img})
                image1 = cv2.imread('img.png')
                image2 = cv2.imread('pred.png')
                image3 = cv2.imread('gt.png')

                # Make sure all images have the same dimensions
                height, width = image1.shape[:2]
                combined_width = width * 3

                # Create a blank canvas for the combined image
                combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)

                # Place the images side by side
                combined_image[:, :width] = image1
                combined_image[:, width:2*width] = image2
                combined_image[:, 2*width:] = image3

                # Save or display the combined image
                
                
                cv2.imwrite('../viz/combined_image_{}.png'.format(i_batch), combined_image)
            
                # Log the image using wandb.Image
                wandb.log({"visualization": wandb.Image(combined_image)})
                




                # print(gt_img.shape)
                
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    
    return "Training Finished!"