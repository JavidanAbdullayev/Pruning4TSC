import numpy as np
import pandas as pd
import sys
import os
import json
import argparse
from distutils.util import strtobool

from utils import load_data, preprocess_data, create_directory, plot_loss_and_acc_curves, InstanceFeatureSparseLoss, CovarianceLoss2

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


from lite_org import LITE

torch.autograd.set_detect_anomaly(True)

import copy
import time 

def get_args():
    
    dataset_names = [ 
                     'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                     'ArrowHead', 'BeetleFly', 'BirdChicken', 'Beef',  'BME', 'Car', 'CBF', 'Chinatown',
                     'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                     'CricketY', 'CricketZ', 
                      'DiatomSizeReduction',
                     'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                     'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
                     'ECG5000', 'ECGFiveDays',  'EOGHorizontalSignal', 'EthanolLevel', 
                     'EOGVerticalSignal',  'FaceAll', 'FaceFour', 'FacesUCR',
                     'FiftyWords', 'Fish', 'FreezerRegularTrain',
                     'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
                     'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
                     'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                     'Ham',  'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                     'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                     'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                     'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                     'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                     'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
                     'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
                     'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                     'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                     'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                     'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
                     'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
                     'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                     'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
                     'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                     'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                     'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'Wine', 'WordSynonyms', 'Yoga', 
                     'UWaveGestureLibraryZ', 'Wafer',  'Worms', 'WormsTwoClass',                     
                     'Crop', 'ElectricDevices',  'FordA', 'FordB', 'HandOutlines',
                     ]
    
    # dataset_names = ['ACSF1', ]
    
    parser = argparse.ArgumentParser(description='Choose to apply which classifier on which dataset with number of runs')

    parser.add_argument('--classifier', help='which classifier to use', type=str, choices=['LITE'], default='LITE', )
    parser.add_argument('--num_filters', help='output directory parent', type=int, default=32,)
    parser.add_argument('--datasets',  help='which dataset to run the experiment on.', type=str, default=dataset_names,)
    parser.add_argument('--runs', help='number of runs to do', type=int, default=1)
    parser.add_argument('--output-directory', help='output directory parent', type=str, default=None)

    args = parser.parse_args()

    return args


def train_alone_model(base_model, epoch):
    print('\n\nTraining epoch: ', epoch)
    base_model.train()
    running_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        outputs, features = base_model.forward_sparsity(inputs.float())
        loss= criterion_CE(outputs, targets) + \
              ins_sparse_loss(features[0][:96]) + ins_sparse_loss(features[1]) + ins_sparse_loss(features[2])
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

    print('Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)'  % (running_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return running_loss.item() / (batch_idx + 1), 100. * correct / total

def test(model):
    print('From the test set evaluation')
    model.eval()
    test_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(valloader):
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs, _= model.forward_sparsity(inputs.float())
        loss = criterion_CE(outputs, targets)
        
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (batch_idx + 1), 100. * correct / total

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


new_seeds = [9409, 3804, 3952, 2561, 5296]
# new_seeds = [9409]

if __name__ == '__main__':
    args = get_args()
    if args.output_directory is None:
        args.output_directory = f'/results/base_models_1e-6_no_weights/'
        
    for dataset in args.datasets:        
        for seed in new_seeds:

            output_directory = os.getcwd() + args.output_directory

            xtrain, ytrain, xtest, ytest = load_data(file_name=dataset)        
            length_TS = int(xtrain.shape[1])
            
            trainloader = preprocess_data(xtrain, ytrain, shuffle=True)
            valloader = preprocess_data(xtest, ytest, shuffle=False)

            output_directory = output_directory + f'ins_sparse_seed_{seed}/' + dataset + '/'
            create_directory(output_directory)

            set_seeds(seed)
            base_model = LITE(output_directory=None, length_TS=length_TS, n_classes=len(np.unique(ytrain)), n_filters=[[32, 32, 32], 32, 32])
            
            first_model_wts = copy.deepcopy(base_model.state_dict())
            torch.save(first_model_wts, output_directory + f'first_model.pt')

            criterion_CE = nn.CrossEntropyLoss()
            ins_sparse_loss = InstanceFeatureSparseLoss(lambda_reg=1e-6)
                        
            optimizer = torch.optim.Adam(base_model.parameters(), )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-4)

            use_cuda = torch.cuda.is_available()
            print('use_cuda: ', use_cuda)
            
            if use_cuda:
                torch.cuda.set_device(0)
                base_model.cuda()
                
            print('length_TS ', length_TS)
            print('Base model performance before training: ')
            val_loss, val_acc = test(base_model)


            min_train_loss = np.inf

            final_loss, learning_rates = [], []
            training_loss, training_acc = [], []
            val_losses, val_acces = [], []

            epochs = 1500
            start_time = time.time()
            best_model_epoch = 0
            
            for epoch in range(epochs):
                train_loss, train_acc = train_alone_model(base_model, epoch)                
                training_loss.append(train_loss)
                training_acc.append(train_acc)

                # val_loss, val_acc = test(base_model)
                # val_losses.append(val_loss)
                # val_acces.append(val_acc)


                if min_train_loss >= train_loss:
                    min_train_loss = train_loss
                    best_model_wts = copy.deepcopy(base_model.state_dict())
                    best_model_epoch = epoch

                scheduler.step(train_loss)
                learning_rates.append(optimizer.param_groups[0]['lr'])
                print('learning rate: ', optimizer.param_groups[0]['lr'])



            last_model_wts = copy.deepcopy(base_model.state_dict())
            torch.save(best_model_wts, output_directory +  'best_model.pt')
            torch.save(last_model_wts, output_directory +  'last_model.pt')










            # Save Logs
            plot_loss_and_acc_curves(training_loss, val_losses, training_acc, val_acces, output_directory)

            duration = time.time() - start_time
            best_model = LITE(output_directory=None, length_TS=length_TS, n_classes=len(np.unique(ytrain)), n_filters=[[32, 32, 32], 32, 32])

            best_model.load_state_dict(best_model_wts)
            best_model.cuda()
            
            print('Best Model Accuracy in below ')
            print('best model epoch: ', best_model_epoch)
            
            start_test_time = time.time()
            test(best_model)
            test_duration = time.time() - start_test_time
            print(test(best_model))
            df = pd.DataFrame(list(zip(training_loss, learning_rates)), columns =['loss', 'learning_rate'])
            index_best_model = df['loss'].idxmin()
            row_best_model = df.loc[index_best_model]
            df_best_model = pd.DataFrame(list(zip([row_best_model['loss']], [index_best_model+1])), columns =['best_model_train_loss', 'best_model_nb_epoch'])

            df.to_csv(output_directory + 'history.csv', index=False)
            df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

            loss_, acc_ = test(best_model)
            df_metrics = pd.DataFrame(list(zip([min_train_loss], [acc_], [duration], [test_duration])), columns =['Loss', 'Accuracy', 'Duration', 'Test Duration'])
            df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False) 
            
                