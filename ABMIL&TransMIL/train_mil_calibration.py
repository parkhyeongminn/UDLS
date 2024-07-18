import enum
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset 
import redis
import pickle
import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve
import random 
import torch.backends.cudnn as cudnn
import json
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from Opt.lookahead import Lookahead
from Opt.radam import RAdam
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram


def load_and_test(test_df, milnet, criterion, optimizer, args, log_path, epoch, model_path):
    state_dict = torch.load(model_path)
    milnet.load_state_dict(state_dict)
    
    test_labels, test_predictions, test_loss, avg_score, auc_value, thresholds_optimal = test(test_df, milnet, criterion, optimizer, args, log_path, epoch)

    print('\n multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))

    return test_labels, test_predictions


def plot_and_save(values, title, filename):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel('Bag Index')
    plt.ylabel(title)
    plt.savefig(filename)
    plt.close()


# random-augment
class BagDataset(Dataset):
    def __init__(self, train_path, args, augment=False) -> None:
        super(BagDataset, self).__init__()
        self.train_path = train_path
        self.args = args
        self.augment = augment

    def get_bag_feats(self, csv_file_df, args):
        if args.dataset.startswith('tcga'):
            feats_csv_path = os.path.join('datasets', 'tcga_imagenet', 'data_tcga_lung_tree', csv_file_df.iloc[0].split('/')[-1] + '.csv')
        else:
            feats_csv_path = csv_file_df.iloc[0]

        df = pd.read_csv(feats_csv_path)
        feats = shuffle(df).reset_index(drop=True)
        feats = feats.to_numpy()
        label = np.zeros(args.num_classes)
        if args.num_classes == 1:
            label[0] = csv_file_df.iloc[1]
        else:
            if int(csv_file_df.iloc[1]) <= (len(label) - 1):
                label[int(csv_file_df.iloc[1])] = 1
        label = torch.tensor(np.array(label))
        feats = torch.tensor(np.array(feats)).float()

        return label, feats

    def augment_feats(self, feats, args):
        dropout_rate = args.dropout_rate
        num_augments = args.num_augments
        augmented_feats = []
        for _ in range(num_augments):
            num_patches = feats.shape[0]
            num_dropout = int(num_patches * dropout_rate)
            keep_indices = np.random.choice(num_patches, num_patches - num_dropout, replace=False)

            augmented = feats[keep_indices]

            augmented_feats.append(augmented)

        return augmented_feats

    def __getitem__(self, idx):
        label, feats = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        if self.augment:
            augmented_feats = self.augment_feats(feats, self.args)
            return label, augmented_feats
        else:
            return label, feats

    def __len__(self):
        return len(self.train_path)
    

def calculate_predictive_entropy(predictions):
    probabilities = torch.sigmoid(predictions)
    mean_probabilities = probabilities.mean(dim=0) 
    log_mean_probabilities = torch.log(mean_probabilities + 1e-12)  # To avoid log(0)
    entropy = -torch.sum(mean_probabilities * log_mean_probabilities)
    return entropy.item()


def calculate_predictive_variance(predictions):
    probabilities = torch.sigmoid(predictions)  
    variance = torch.var(probabilities, dim=0)
    mean_variance = variance.mean().item()
    return mean_variance


def scale_entropy(entropies, args):
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    scaled_entropies = [(e - min_entropy) / (max_entropy - min_entropy + 1e-12) * args.smoothing_factor for e in entropies]
    return scaled_entropies


# random-augmented train
def train(train_df, milnet, criterion, optimizer, args, log_path):
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    entropies = []
    variances = []
    uncertainties = []
    
    for i,(bag_label, bag_feats_list) in enumerate(train_df):
        bag_label = bag_label.cuda()
        losses = []
        predictions = []

        for bag_feats in bag_feats_list:
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim
            #print(bag_feats.shape)
            optimizer.zero_grad()

            if args.model == 'abmil':
                bag_prediction, _, attention = milnet(bag_feats)
                loss =  criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))

            elif args.model == 'transmil':
                output = milnet(bag_feats)
                bag_prediction, bag_feature ,attention=  output['logits'], output["Bag_feature"], output["A"]
                loss =  criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
  
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            predictions.append(bag_prediction.cpu().detach())

            if args.c_path:
                attention = output['A']
                atten_max = atten_max+ attention.max().item()
                atten_min = atten_min+attention.min().item()
                atten_mean = atten_mean+ attention.mean().item()
                sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f attention max:%.4f, min:%.4f, mean:%.4f' 
                % (i, len(train_df), loss.item(), attention.max().item(), attention.min().item(), attention.mean().item()))
            
            else:
                sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f ' % (i, len(train_df), loss.item()))

        avg_loss = sum(losses) / len(losses)
        total_loss += avg_loss

        predictions = torch.stack(predictions)

        predictive_entropy = calculate_predictive_entropy(predictions)
        predictive_variance = calculate_predictive_variance(predictions)
        uncertainty = predictive_entropy * predictive_variance
        entropies.append(predictive_entropy)
        variances.append(predictive_variance)
        uncertainties.append(uncertainty)

        sys.stdout.write('\r Training bag [%d/%d] avg bag loss: %.4f, entropy: %.4f, variance: %.4f, uncertainty: %.4f' %
                         (i, len(train_df), avg_loss, predictive_entropy, predictive_variance, uncertainty))

    if args.c_path:
        atten_max = atten_max / len(train_df)
        atten_min =  atten_min / len(train_df)
        atten_mean = atten_mean / len(train_df)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n atten_max'+str(atten_max))
                log_txt.write('\n atten_min'+str(atten_min))
                log_txt.write('\n atten_mean'+str(atten_mean))

    return total_loss / len(train_df), entropies, variances, uncertainties


# train with label smoothing
def train_with_smoothing(train_df, milnet, criterion, optimizer, args, log_path, epoch, entropies):
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0

    for i, (bag_label, bag_feats) in enumerate(train_df):
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)
        optimizer.zero_grad()

        if args.model == 'abmil':
            bag_prediction, _, attention = milnet(bag_feats)
            loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))

        elif args.model == 'transmil':
            output = milnet(bag_feats)
            bag_prediction, bag_feature ,attention=  output['logits'], output["Bag_feature"], output["A"]
            loss =  criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))

        smoothing = entropies[i]
        smoothed_label = (1 - smoothing) * bag_label + smoothing / args.num_classes
        loss = criterion(bag_prediction.view(1, -1), smoothed_label.view(1, -1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if args.c_path:
            attention = output['A']
            atten_max = atten_max+ attention.max().item()
            atten_min = atten_min+attention.min().item()
            atten_mean = atten_mean+ attention.mean().item()
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f attention max:%.4f, min:%.4f, mean:%.4f' 
            % (i, len(train_df), loss.item(), attention.max().item(), attention.min().item(), attention.mean().item()))
           
        else:
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f ' % (i, len(train_df), loss.item()))

    if args.c_path:
        atten_max = atten_max / len(train_df)
        atten_min =  atten_min / len(train_df)
        atten_mean = atten_mean / len(train_df)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n atten_max'+str(atten_max))
                log_txt.write('\n atten_min'+str(atten_min))
                log_txt.write('\n atten_mean'+str(atten_mean))

    return total_loss / len(train_df)

# random-augment test
def test(test_df, milnet, criterion, optimizer, args, log_path, epoch):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i,(bag_label,bag_feats) in enumerate(test_df):
            label = bag_label.numpy()
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)

            if args.model in ['abmil', 'max', 'mean']:
                bag_prediction, _, _ = milnet(bag_feats)
                max_prediction = bag_prediction
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model == 'transmil':
                output = milnet(bag_feats)
                bag_prediction, bag_feature =  output['logits'], output["Bag_feature"]
                max_prediction = bag_prediction
                loss =  criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend(label)
            if args.average:  # notice args.average here
                test_predictions.extend([(0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([(0.0 * torch.sigmoid(max_prediction) + 1.0 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    test_predictions1 = test_predictions.copy()

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    with open(log_path, 'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print('\n')
        print(confusion_matrix(test_labels, test_predictions))
        info = confusion_matrix(test_labels, test_predictions)
        with open(log_path, 'a') as log_txt:
            log_txt.write('\n' + str(info))
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            print(confusion_matrix(test_labels[:, i], test_predictions[:, i]))
            info = confusion_matrix(test_labels[:, i], test_predictions[:, i])
            with open(log_path, 'a') as log_txt:
                log_txt.write('\n' + str(info))
    bag_score = 0
    # average acc of all labels
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)  # ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)

    print('\n  multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100, sum(auc_value) / len(auc_value) * 100))
    print('\n', cls_report)
    with open(log_path, 'a') as log_txt:
        log_txt.write('\n multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100, sum(auc_value) / len(auc_value) * 100))
        log_txt.write('\n' + cls_report)

    return test_labels, test_predictions1, total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label)==0:
            continue
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Train UDLS for abmil and transmil')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [1024]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=2, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='transmil', type=str, help='MIL model [admil, transmil]')
    parser.add_argument('--smoothing_factor', default=0.1, type=float, help='Label smoothing factor [0]')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--agg', type=str,help='which agg')
    parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
    parser.add_argument('--c_learn', action='store_true', help='learn confounder or not')
    parser.add_argument('--c_dim', default=128, type=int, help='Dimension of the projected confounders')
    parser.add_argument('--freeze_epoch', default=999, type=int, help='freeze confounders during this many epoch from the start')
    parser.add_argument('--c_merge', type=str, default='cat', help='cat or add or sub')
    parser.add_argument('--n_bins', type=int, default=10, help='the number of bins')
    parser.add_argument('--num_augments', type=int, default=2, help='the number of augments')
    args = parser.parse_args()

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.c_path:
        save_path = os.path.join('deconf', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_c_path')
    else:
        save_path = os.path.join('baseline', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_'
                                 +str(args.smoothing_factor)+'_'+str(args.dropout_rate)+'_fulltune')
        
    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file,'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'
    
    # seed 
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    '''
    model 
    1. set require_grad    
    2. choose model and set the trainable params 
    3. load init
    '''

    if args.model == 'abmil':
        import abmil as mil
        milnet = mil.Attention(in_size=args.feats_size, out_size=args.num_classes,confounder_path=args.c_path, \
            confounder_learn=args.c_learn, confounder_dim=args.c_dim, confounder_merge=args.c_merge).cuda()
    elif args.model == 'transmil':
        import Models.TransMIL.net as mil
        milnet = mil.TransMIL(input_size=args.feats_size, n_classes=args.num_classes, confounder_path=args.c_path).cuda()

    for name, _ in milnet.named_parameters():
        print('Training {}'.format(name))
        with open(log_path,'a+') as log_txt:
            log_txt.write('\n Training {}'.format(name))

    if args.dataset.startswith("tcga"):
        bags_csv = 'datasets_csv/TCGA.csv'
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
        test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]

    elif args.dataset.startswith('C'):
        bags_csv = 'datasets_csv/Camelyon16.csv'
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:270, :]
        test_path = bags_path.iloc[270:, :]

    trainset = BagDataset(train_path, args, augment=True)
    train_loader = DataLoader(trainset, 1, shuffle=True, num_workers=4)
    original_trainset = BagDataset(train_path, args, augment=False)
    original_train_loader = DataLoader(original_trainset, 1, shuffle=True, num_workers=4)
    testset = BagDataset(test_path, args, augment=False)
    test_loader = DataLoader(testset, 1, shuffle=False, num_workers=4)

    print('len train_loader: ', len(train_loader))
    print('len original_train_loader: ', len(original_train_loader))

    # sanity check begins here
    print('*******sanity check *********')
    for k, v in milnet.named_parameters():
        if v.requires_grad == True:
            print(k)

    # loss, optim, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()), 
                                lr=args.lr, betas=(0.5, 0.9), 
                                weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    best_score = 0

    if args.test:
        epoch = args.num_epochs-1
        test_labels, test_predictions, test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)   
                
        train_loss_bag = 0
        if args.dataset=='TCGA-lung':
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        
        sys.exit()
        
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        train_loss_bag, entropies, variances, uncertainties = train(train_loader, milnet, criterion, optimizer, args, log_path) # iterate all bags
        print('Epoch time:{}'.format(time.time() - start_time))
        test_labels, test_predictions1, test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
        
        info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: '%(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)) + '\n'
        with open(log_path, 'a+') as log_txt:
            log_txt.write(info)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
              (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))

        scheduler.step()
        # current_score = (sum(aucs) + avg_score) / 2
        # if current_score >= best_score:
        #     best_score = current_score
        #     save_name = os.path.join(save_path, str(run+1) + '.pth')
        #     torch.save(milnet.state_dict(), save_name)
        #     with open(log_path, 'a+') as log_txt:
        #         info = 'Best model saved at: ' + save_name + '\n'
        #         log_txt.write(info)
        #         info = 'Best thresholds ===>>> ' + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)) + '\n'
        #         log_txt.write(info)
        #     print('Best model saved at: ' + save_name)
        #     print('Best thresholds ===>>> ' + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
        # if epoch == args.num_epochs:
        #     save_name = os.path.join(save_path, 'last.pth')
        #     torch.save(milnet.state_dict(), save_name)
    log_txt.close()

    # Plot and save predictive entropy and variance
    plot_and_save(entropies, 'Predictive Entropy', os.path.join(save_path, 'predictive_entropy.png'))
    plot_and_save(variances, 'Predictive Variance', os.path.join(save_path, 'predictive_variance.png'))
    plot_and_save(uncertainties, 'Uncertainty', os.path.join(save_path, 'uncertainty.png'))

    scaled_entropies = scale_entropy(variances, args)

    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        train_loss_bag = train_with_smoothing(original_train_loader, milnet, criterion, optimizer, args, log_path, epoch - 1, entropies=scaled_entropies)  # iterate all bags
        print('Epoch time:{}'.format(time.time() - start_time))
        test_labels, test_predictions1, test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
        
        info = 'Epoch with smoothing [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: '%(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)) + '\n'
        with open(log_path, 'a+') as log_txt:
            log_txt.write(info)
        print('\r Epoch with smoothing [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
              (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))

        scheduler.step()
        current_score = (sum(aucs) + avg_score) / 2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1) + '.pth')
            torch.save(milnet.state_dict(), save_name)
            with open(log_path, 'a+') as log_txt:
                info = 'Best model saved at: ' + save_name + '\n'
                log_txt.write(info)
                info = 'Best thresholds ===>>> ' + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)) + '\n'
                log_txt.write(info)
            print('Best model saved at: ' + save_name)
            print('Best thresholds ===>>> ' + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
        if epoch == args.num_epochs:
            save_name = os.path.join(save_path, 'last.pth')
            torch.save(milnet.state_dict(), save_name)
    log_txt.close()

    if os.path.exists(save_name):
        print("Loading best model from: ", save_name)
        test_labels, test_predictions = load_and_test(test_loader, milnet, criterion, optimizer, args, log_path, epoch, save_name)

    ece = ECE(args.n_bins)
    ece_score = ece.measure(test_predictions, test_labels)
    print(f'ECE: {ece_score:.4f}')

    diagram = ReliabilityDiagram(args.n_bins)
    diagram.plot(test_predictions, test_labels)
    diagram_filename = f'ReliabilityDiagram_{args.model}_{args.dataset}_{args.smoothing_factor}_{args.dropout_rate}_{args.num_augments}_variance.png'
    plt.savefig(diagram_filename)  # Save the reliability diagram
    plt.close()


if __name__ =='__main__':
    main()