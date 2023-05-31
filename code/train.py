# Imports
import os
import copy
import pandas as pd
import numpy as np
import csv
import logging
from tqdm import tqdm
from pathlib import Path

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# Project Imports
from data_utils import FaceDataset
from metrics_utils import performances_compute, performances_compute2
from resnet_utils_min import Resnet18_Min, Resnet34_Min
from resnet_utils_mult import Resnet18_Mult, Resnet34_Mult



# Function: Train Function
def train_fn(model, data_loader, data_size, optimizer, criterion, weight_loss, loss_measure, autoenc_bf_path, autoenc_morph_path, lmbda, device,epoch):
    model.train()

    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_corrects = 0
    running_loss_norm = 0.0
    running_loss_similarity = 0.0

    cos_sim = torch.nn.CosineSimilarity(dim=2)

    for i, (inputs, labels, lv_1, lv_2) in enumerate(tqdm(data_loader)):
        inputs, labels, lv_1, lv_2 = inputs.to(device), torch.FloatTensor(labels *1.0).to(device), lv_1.to(device), lv_2.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if loss_measure == 'ortho':
            loss_1 = criterion(outputs[2], labels)
            loss_2 = weight_loss * torch.bmm(outputs[0].view(outputs[2].shape[0], 1, -1), outputs[1].view(outputs[2].shape[0], -1, 1)).reshape(outputs[2].shape[0]).pow(2).mean()
            loss =  loss_1+loss_2 
            running_loss_1 += loss_1.item() * inputs.size(0)
            running_loss_2 += loss_2.item() * inputs.size(0)
        elif loss_measure == 'bce':
            loss = criterion(outputs[2], labels)
        elif loss_measure == 'kd':
            loss_1 = criterion(outputs[2], labels)    
            loss_2 = weight_loss * (-(1 + lmbda * torch.norm(outputs[1].view(outputs[2].shape[0], 1, -1), p=2)) * torch.div(labels, cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze()) + torch.mul(1 - labels, (cos_sim(lv_1, lv_2).squeeze() - cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), outputs[1].view(outputs[2].shape[0], 1, -1)).squeeze()).pow(2))).mean()
            loss_norm = -(1 + lmbda * torch.norm(outputs[1].view(outputs[2].shape[0], 1, -1), p=2)) * torch.div(labels, cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze()).mean()
            loss_similarity = torch.mul(1 - labels, (cos_sim(lv_1, lv_2).squeeze() - cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), outputs[1].view(outputs[2].shape[0], 1, -1)).squeeze()).pow(2)).mean()
            #torch.mul(labels, cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze() + lmbda * torch.norm(outputs[1].view(outputs[2].shape[0], 1, -1), p=2)) 
            loss =  loss_1+loss_2 
            running_loss_1 += loss_1.item() * inputs.size(0)
            running_loss_2 += loss_2.item() * inputs.size(0)
            running_loss_norm += loss_norm.item() * inputs.size(0)
            running_loss_similarity += loss_similarity.item() * inputs.size(0)
        elif loss_measure == 'altered_kd':
            loss_1 = criterion(outputs[2], labels)    
            loss_2 = weight_loss * ((1 + lmbda * torch.norm(outputs[1].view(outputs[2].shape[0], 1, -1), p=2)) * torch.div(labels, torch.abs(cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze())) + torch.mul(1 - labels, (cos_sim(lv_1, lv_2).squeeze() - cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), outputs[1].view(outputs[2].shape[0], 1, -1)).squeeze()).pow(2))).mean()
            loss_norm = (1 + lmbda * torch.norm(outputs[1].view(outputs[2].shape[0], 1, -1), p=2)) * torch.div(labels, torch.abs(cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze())).mean()
            loss_similarity = torch.mul(1 - labels, (cos_sim(lv_1, lv_2).squeeze() - cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), outputs[1].view(outputs[2].shape[0], 1, -1)).squeeze()).pow(2)).mean()
            loss =  loss_1+loss_2 
            running_loss_1 += loss_1.item() * inputs.size(0)
            running_loss_2 += loss_2.item() * inputs.size(0)
            running_loss_norm += loss_norm.item() * inputs.size(0)
            running_loss_similarity += loss_similarity.item() * inputs.size(0)
        elif loss_measure == 'prefusion':
            loss_1 = criterion(outputs[2], labels)
            index_similarity = torch.abs(cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze()) < torch.abs(cos_sim(outputs[1].view(outputs[2].shape[0], 1, -1), lv_1).squeeze())
            #
            part1 = np.minimum(int(epoch/20),1) * (1-outputs[4].pow(2) + (outputs[3]).pow(2)) - torch.abs(cos_sim(outputs[1].view(outputs[2].shape[0], 1, -1), lv_1).squeeze())
            part2 = np.minimum(int(epoch/20),1) * (1-outputs[3].pow(2) + (outputs[4]).pow(2)) - torch.abs(cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze())

            alternative_term_bf = torch.where((torch.abs(cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze()) < torch.abs(cos_sim(outputs[1].view(outputs[2].shape[0], 1, -1), lv_1).squeeze())),part1,part2) 
            loss_2 = weight_loss * ((alternative_term_bf * labels) + torch.mul(1 - labels, (cos_sim(lv_1, lv_2).squeeze() - cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), outputs[1].view(outputs[2].shape[0], 1, -1)).squeeze()).pow(2))).mean()
            
            loss_norm = ((alternative_term_bf * labels).squeeze()).mean()
            loss_similarity = torch.mul(1 - labels, (cos_sim(lv_1, lv_2).squeeze() - cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), outputs[1].view(outputs[2].shape[0], 1, -1)).squeeze()).pow(2)).mean()
            #torch.mul(labels, cos_sim(outputs[0].view(outputs[2].shape[0], 1, -1), lv_1).squeeze() + lmbda * torch.norm(outputs[1].view(outputs[2].shape[0], 1, -1), p=2)) 
            loss =  loss_1+loss_2 
            running_loss_1 += loss_1.item() * inputs.size(0)
            running_loss_2 += loss_2.item() * inputs.size(0)
            running_loss_norm += loss_norm.item() * inputs.size(0)
            running_loss_similarity += loss_similarity.item() * inputs.size(0)


        _, preds = torch.max(outputs[2].reshape((-1,1)),dim=1)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
       
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / data_size

    epoch_acc = running_corrects.double() / data_size
    if loss_measure != 'bce':
        epoch_loss_1 = running_loss_1 / data_size
        epoch_loss_2 = running_loss_2 / data_size
        print('{} Loss: {:.4f} Loss C: {:.4f} Loss extra: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss,epoch_loss_1,epoch_loss_2, epoch_acc))
        if loss_measure != 'ortho':
            print('{} Loss norm: {:.4f} Loss similarity: {:.4f}'.format('Train', running_loss_norm / data_size, running_loss_similarity / data_size))
    else:
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc



# Function: Evaluation Function
def eval_fn(model, data_loader, data_size, criterion, device):
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0

        prediction_scores, gt_labels = [], []
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), torch.FloatTensor(labels *1.0).to(device)

            outputs = model(inputs)
            loss= criterion(outputs[2], labels)

            _, preds = torch.max(outputs[2].reshape((-1,1)),dim=1)

            probs = outputs[2].reshape((-1,1))
            for i in range(probs.shape[0]):
                prediction_scores.append(float(probs[i].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # import pdb
        # pdb.set_trace()
        
        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects.double() / data_size
        auc, eer_value, _ = performances_compute(prediction_scores, gt_labels, verbose=False)
        _, bpcer01, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.001, verbose=False)
        _, bpcer1, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.01, verbose=False)
        _, bpcer10, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.1, verbose=False)
        _, bpcer20, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.2, verbose=False)


        print('{} Loss: {:.4f} Auc: {:.4f} EER: {:.4f} MinP {:.4f} MaxP {:.4f}'.format('Val', epoch_loss, auc, eer_value,min(prediction_scores),max(prediction_scores)))
        print('{} BPCER 0.1: {:.4f} BPCER 1.0: {:.4f} BPCER 10.0 {:.4f} BPCER 20.0 {:.4f}'.format('val',bpcer01,bpcer1,bpcer10,bpcer20))
    
    return epoch_loss, epoch_acc, eer_value, bpcer20, bpcer10, bpcer1, bpcer01



# Function: Run training
def run_training(model, model_path, device, logging_path, num_epochs, dataloaders, dataset_sizes, lr, weight_loss, loss_measure, output_name, earlystop_patience=35):
    model = model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer=optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    logging.basicConfig(filename=logging_path, level=logging.INFO)

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_eer = 100
    lowest_20 = 1000
    lowest_10 = 1000
    lowest_1 = 1000
    lowest_01 = 1000
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)
        # Each epoch has a training and validation phase
        train_loss, train_acc = train_fn(model, dataloaders['train'], dataset_sizes['train'], optimizer, criterion, weight_loss, loss_measure, args.autoenc_bf_path, args.autoenc_morph_path, args.lmbda, device=device,epoch=epoch)
        val_loss, val_acc, val_eer_values,out_20,out_10,out_1,out_01 = eval_fn(model, dataloaders['val'], dataset_sizes['val'], criterion, device=device)
        logging.info('train loss: {}, train acc: {}, val loss: {}, val acc: {}, val eer: {}'.format(train_loss, train_acc, val_loss, val_acc, val_eer_values))

        if epoch % 5 == 0:
            mean = run_extra_test(model, device)
        # deep copy the model
            if mean < lowest_eer:
                lowest_eer = mean
                lowest_20 = out_20
                lowest_10 = out_10
                lowest_1 = out_1
                lowest_01 = out_01
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
        # if val_eer_values <= lowest_eer:
        #     lowest_eer = val_eer_values
        #     lowest_20 = out_20
        #     lowest_10 = out_10
        #     lowest_1 = out_1
        #     lowest_01 = out_01
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == earlystop_patience or epoch >= num_epochs:
            early_stop = True
        else:
            continue

        if early_stop:
            print('Train process Stopped')
            print('epoch: {}'.format(epoch))
            break

    print('Lowest EER: {:4f}'.format(lowest_eer))
    logging.info('Lowest EER: {:4f}'.format(lowest_eer))
    logging.info(f'saved model path: {model_path}')

    with open("results_"+output_name, mode='a') as csv_file:
        fieldnames = ['lower_eer','01','1','10','20']
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        writer.writerow([lowest_eer,lowest_01,lowest_1,lowest_10,lowest_20])

    # save best model weights
    torch.save(best_model_wts, model_path)



# Function: Run test
def run_test(test_loader, model, model_path, device):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prediction_scores, gt_labels = [], []
    with torch.no_grad():
        running_corrects = 0

        for inputs, labels in tqdm(test_loader):
            inputs, labels= inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = outputs[2]

            for i in range(probs.shape[0]):
                prediction_scores.append(float(probs[i].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))

        std_value = np.std(prediction_scores)
        mean_value = np.mean(prediction_scores)
        prediction_scores = [ (float(i) - mean_value) /(std_value) for i in prediction_scores]

        auc, eer_value, _ = performances_compute(prediction_scores, gt_labels, verbose=False)
        _, bpcer01, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.001, verbose=False)
        _, bpcer1, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.01, verbose=False)
        _, bpcer10, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.1, verbose=False)
        _, bpcer20, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.2, verbose=False)

        print(f'Test auc value: {auc*100}')
        print(f'Test EER value: {eer_value*100}')
        print(f'Test BPCER_0.1 value: {bpcer01*100}')
        print(f'Test BPCER_1 value: {bpcer1*100}')
        print(f'Test BPCER_10 value: {bpcer10*100}')
        print(f'Test BPCER_20 value: {bpcer20*100}')

    return prediction_scores



# Function: Write scores to a .CSV file
def write_scores(test_csv, prediction_scores, output_path):
    save_data = []
    dataframe = pd.read_csv(test_csv)
    for idx in range(len(dataframe)):
        image_path = dataframe.iloc[idx, 0]
        label = dataframe.iloc[idx, 1]
        label = label.replace(' ', '')
        save_data.append({'image_path': image_path, 'label':label, 'prediction_score': prediction_scores[idx]})

    with open(output_path, mode='w') as csv_file:
        fieldnames = ['image_path', 'label', 'prediction_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in save_data:
            writer.writerow(data)

    print(f'Saving prediction scores in {output_path}.\n')


def run_extra_test(model,device): 
    model = model.to(device)
    model.eval()
    mean = 0
    for test_csv_path in ["/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/facelab_london/asml.csv","/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/facelab_london/stylegan.csv","/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/facelab_london/facemorpher.csv","/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/facelab_london/opencv.csv","/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/facelab_london/webmorph.csv"]:
        print(f"Generating test results on: {test_csv_path}")
        # Create the test set
        test_dataset = FaceDataset(
            file_name=test_csv_path,
            split="test",
            latent_size=args.latent_size
        )
        
        # Create the test loader
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        
        # Add the test loader and the dataset size to corresponding dictionaries
        dataloaders = {'test': test_loader}
        dataset_sizes = {'test': len(test_dataset)}
        print('Test length:', len(test_dataset))
        
        prediction_scores, gt_labels = [], []
        with torch.no_grad():
            running_corrects = 0

            for inputs, labels in tqdm(test_loader):
                inputs, labels= inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = outputs[2]

                for i in range(probs.shape[0]):
                    prediction_scores.append(float(probs[i].detach().cpu().numpy()))
                    gt_labels.append(int(labels[i].detach().cpu().numpy()))

            std_value = np.std(prediction_scores)
            mean_value = np.mean(prediction_scores)
            prediction_scores = [ (float(i) - mean_value) /(std_value) for i in prediction_scores]

            auc, eer_value, _ = performances_compute(prediction_scores, gt_labels, verbose=False)
            _, bpcer1, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.01, verbose=False)
            _, bpcer20, _ = performances_compute2(prediction_scores, gt_labels,threshold_type='bpcer', op_val=0.2, verbose=False)

            print(f'Test auc value: {auc*100}')
            print(f'Test EER value: {eer_value*100}')
            print(f'Test BPCER_1 value: {bpcer1*100}')
            print(f'Test BPCER_20 value: {bpcer20*100}')
            mean += eer_value
    return mean/5


# Function: Main
def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    model_name = args.model_name.lower()

    assert model_name in ("resnet18_min", "resnet34_min", "resnet18_mult", "resnet34_mult"), "Please provide a valid model name ('resnet18_min', 'resnet34_min', 'resnet18_mult', 'resnet34_mult')-"


    # Erase uppon review
    # models = ["resnet18"]
    # for model_name in models:
    
    
    if model_name == "resnet18_min":
        model = Resnet18_Min(args.latent_size)
    elif model_name == "resnet34_min":
        model = Resnet34_Min(args.latent_size)
    elif model_name == "resnet18_mult":
        model = Resnet18_Mult(args.latent_size)
    elif model_name == "resnet34_mult":
        model = Resnet34_Mult(args.latent_size)
    
    # Print model name
    # print(model)
    print(f"Model: {model_name} with embedding latent size: {args.latent_size}.")
        
    if args.is_train:

        # Create the train set
        train_dataset = FaceDataset(
            file_name=args.train_csv_path,
            split="train",
            latent_size=args.latent_size
        )
        
        # Create the validation set
        val_dataset = FaceDataset(
            file_name=args.train_csv_path,
            split="validation",
            latent_size=args.latent_size
        )
        

        # Create the dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        # Add dataloaders to a dictionary
        dataloaders = {'train': train_loader, 'val': val_loader}
        
        # Add the sizes of these datasets to a dictionary
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
        print('Train and Validation lengths:', len(train_dataset), len(val_dataset))

        # compute loss weights to improve the unbalance between data
        attack_num, bonafide_num = 0, 0
        with open(args.train_csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['label'] == 'attack':
                    attack_num += 1
                else:
                    bonafide_num += 1
        print('attack and bonafide num:', attack_num, bonafide_num)

        # nSamples  = [attack_num, bonafide_num]
        # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        # normedWeights = torch.FloatTensor(normedWeights).to(device)

        #create log file and train model
        logging_path = os.path.join(args.output_dir, 'train_info.log')
        run_training(model, args.model_path, device, logging_path, args.max_epoch, dataloaders, dataset_sizes,args.lr,args.weight_loss, args.loss_measure, output_name=model_name)
    else:
        #loading the model in case it is already trained
        model.load_state_dict(torch.load(args.model_path))

    if args.is_test:

        # Print the name of the test set
        print(f"Generating test results on: {args.test_csv_path}")
        # Create the test set
        test_dataset = FaceDataset(
            file_name=args.test_csv_path,
            split="test",
            latent_size=args.latent_size
        )
        
        # Create the test loader
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        
        # Add the test loader and the dataset size to corresponding dictionaries
        dataloaders = {'test': test_loader}
        dataset_sizes = {'test': len(test_dataset)}
        print('Test length:', len(test_dataset))
        
        
        # create save directory and path
        test_output_folder = os.path.join(args.output_dir, 'test_results')
        Path(test_output_folder).mkdir(parents=True, exist_ok=True)
        test_output_path = os.path.join(test_output_folder, 'test_results.csv')
        # test
        test_prediction_scores = run_test(test_loader=test_loader, model=model, model_path=args.model_path, device=device)
        write_scores(args.test_csv_path, test_prediction_scores, test_output_path)



# Main
if __name__ == '__main__':

    cudnn.benchmark = True

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.manual_seed(0)
    else:
        print('GPU is not available')
        torch.manual_seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='MixFaceNet model')
    parser.add_argument("--train_csv_path", default="mor_gan_train.csv", type=str, help="input path of train csv")
    parser.add_argument("--test_csv_path", default="mor_gan_train.csv", type=str, help="input path of test csv")

    parser.add_argument("--model_name", choices=["resnet18_min", "resnet34_min", "resnet18_mult", "resnet34_mult"], type=str, required=True, help="Model name: 'resnet18_min', 'resnet34_min', 'resnet18_mult' or 'resnet34_mult')")

    parser.add_argument("--output_dir", default="output", type=str, help="path where trained model and test results will be saved")
    parser.add_argument("--model_path", default="mixfacenet_SMDD", type=str, help="path where trained model will be saved or location of pretrained weight")

    parser.add_argument("--is_train", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="train database or not")
    parser.add_argument("--is_test", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="test database or not")

    parser.add_argument("--max_epoch", default=1, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="train batch size")
    parser.add_argument("--latent_size", default=64, type=int, help="train batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="train batch size")
    parser.add_argument("--weight_loss", default=1, type=float, help="first regularization factor")
    parser.add_argument("--loss_measure", default="ortho", type=str, help="bce ortho KD")
    parser.add_argument("--autoenc_bf_path", type=str, help="path to autoencoder latent vectors (bonafide images)")
    parser.add_argument("--autoenc_morph_path", type=str, help="path to autoencoder latent vectors (bonafide images used to generate morphing attacks)")
    parser.add_argument("--lmbda", default=1, type=float, help="second regularization factor")
    
    parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU.")

    args = parser.parse_args()

    main(args)
