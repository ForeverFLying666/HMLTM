
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import pickle
import shutil
from util.dataset import Rumor_Data
from util.indicator import precision_true, recall_true, f1_true, precision_fake, recall_fake, f1_fake
from util.util import to_var, to_np, init, set_seed
from conf.argu import parse_arguments
from model.core import CNN_Fusion, set_dropout_to_eval, set_dropout_to_train


def train_main(args, train_loader, learning_rate, batch_size, num_epochs, ratio, criterion, patience,
               temp_file_name):
    model = CNN_Fusion(args, ratio, ratio)
    init(model)
    set_dropout_to_train(model)
    model.train()
    external_memory_bank = torch.randn(3, 3*768).to(int(torch.cuda.current_device()))
    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()
    best_validate_acc = 0.000
    best_validate_dir = ''
    t_c = 0.5
    t_d = 0.5
    t_c_best = 0.5
    t_d_best = 0.5
    mean_best = 0
    std_best = 0
    mean_list = []
    std_list = []
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.0003)
    MILESTONES = [20,30]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
    print('begin training model')
    for epoch in range(num_epochs):
        print(external_memory_bank)
        if epoch == 19:
            t_c = t_c + 0.5
            t_d = t_d + 0.5
        if epoch == 29:
            t_c = t_c + 1
            t_d = t_d + 1
        class_cost_vector = []
        acc_vector = []
        train_mean = 0
        train_std = 0
        for i, (train_data, train_labels) in enumerate(train_loader):
            first_text_embedding, middle_text_embedding, last_text_embedding, \
            train_text_simi_image_feature, train_similarity, \
            train_image, train_labels = \
                to_var(train_data[0]), \
                to_var(train_data[1]), \
                to_var(train_data[2]), \
                to_var(train_data[3]), \
                to_var(train_data[4]), \
                to_var(train_data[5]), \
                to_var(train_labels)
            optimizer.zero_grad()
            train_mean = torch.mean(train_similarity, dim=0)
            train_std = torch.std(train_similarity, dim=0)
            train_similarity = (train_similarity - train_mean) / train_std

            class_outputs, external_memory_bank = model(first_text_embedding, middle_text_embedding,
                                                        last_text_embedding, train_text_simi_image_feature,
                                                        train_similarity, train_image, t_c, t_d, "train",
                                                        external_memory_bank)

            # Class Loss
            class_loss = criterion(class_outputs, torch.as_tensor(train_labels, dtype=torch.long))
            class_loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)
            accuracy = (train_labels == argmax.squeeze()).float().mean()
            class_cost_vector.append(class_loss.item())
            acc_vector.append(accuracy.item())
            mean_list.append(train_mean.item())
            std_list.append(train_std.item())
        scheduler.step()

        with open(temp_file_name, "a", encoding="utf8") as f:
            f.write("Epoch:" + str(epoch + 1) +
                    "   Train Loss:" + str(np.mean(class_cost_vector)) +
                    "   Train_Acc:" + str(np.mean(acc_vector)) + "\n")
        if np.mean(acc_vector) > best_validate_acc:
            best_validate_acc = np.mean(acc_vector)
            best_validate_dir = args.output_file + "CNN_Fusion_" \
                                + str(learning_rate) + "_" \
                                + str(batch_size) + "_" \
                                + str(num_epochs) + "_" \
                                + str(ratio) + "_" \
                                + str(epoch + 1) + '.pkl'
            if not os.listdir(args.output_file):
                torch.save(model.state_dict(), best_validate_dir)
            else:
                shutil.rmtree(args.output_file)
                os.mkdir(args.output_file)
                torch.save(model.state_dict(), best_validate_dir)

            t_c_best = t_c
            t_d_best = t_d
            mean_best = train_mean
            std_best = train_std
            best_external_memory_bank = external_memory_bank

    return best_validate_dir, np.mean(mean_list), np.mean(std_list), t_c_best, t_d_best, external_memory_bank


def test_main(args, test_loader, batch_size, ratio, best_validate_dir, mean_best, std_best, t_c_best, t_d_best,
              temp_file_name, external_memory_bank):
    """========================================Test the Model============================================"""
    model = CNN_Fusion(args, ratio, ratio)
    model.load_state_dict(torch.load(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()

    set_dropout_to_eval(model)
    model.eval()

    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels) in enumerate(test_loader):
        first_text_embedding, \
        middle_text_embedding, \
        last_text_embedding, \
        text_simi_image_feature, \
        test_similarity, \
        test_image, \
        test_labels = to_var(test_data[0]), to_var(test_data[1]), to_var(test_data[2]), \
                      to_var(test_data[3]), to_var(test_data[4]), to_var(test_data[5]), to_var(test_labels)

        test_similarity = (test_similarity - mean_best) / std_best

        test_outputs, external_memory_bank = model(first_text_embedding,
                                                   middle_text_embedding,
                                                   last_text_embedding,
                                                   text_simi_image_feature,
                                                   test_similarity,
                                                   test_image, 1, 1, "test", external_memory_bank)

        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs)), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_true = list(test_true)
    test_pred = list(test_pred)

    with open(temp_file_name, "a", encoding="utf8") as f:
        f.write("Classification Acc" + str(test_accuracy) +
                "   F1:" + str(test_f1) +
                "   Precision:" + str(test_precision) +
                "   Recall:" + str(test_recall) +
                "   AUC-ROC:" + str(test_aucroc) + "\n"
                "   real-p:" + str(precision_true(test_pred, test_true)) +
                "   real-r:" + str(recall_true(test_pred, test_true)) +
                "   real-f:" + str(f1_true(test_pred, test_true)) +
                "   fake-p:" + str(precision_fake(test_pred, test_true)) +
                "   fake-r:" + str(recall_fake(test_pred, test_true)) +
                "   fake-f:" + str(f1_fake(test_pred, test_true)) + "\n")

        f.write(best_validate_dir + "\n\n\n")


def main(args):
    print("begin loading data")
    with open("g-train.pkl", "rb") as f1:
        temp = pickle.load(f1)
        train_dataset_temp = {"input_ids": temp['input_ids'],
                              "attention_mask": temp['attention_mask'],
                              "token_type_ids": temp['token_type_ids'],
                              "text_simi_image_feature": temp['text_simi_image_feature'],
                              "image": temp['image'],
                              "label": temp['label'],
                              "similarity": temp['similarity']}

    print("finish loading data")

    print("begin create dataset")
    train_dataset = Rumor_Data(train_dataset_temp)
    print("finish create dataset")

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.0001
    batch_size = 32
    num_epochs = 2
    patience = 7
    ratio = 0.0

    print('begin building model')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    temp_file_name = "result\\" \
                     + "_learning_rate=" + str(learning_rate) \
                     + "_ratio=" + str(ratio) \
                     + "_patience=" + str(patience) \
                     + ".txt"

    best_validate_dir, mean_best, std_best, t_c_best, t_d_best, external_memory_bank = train_main(args, train_loader,
                                                                                                  learning_rate,
                                                                                                  batch_size,
                                                                                                  num_epochs,
                                                                                                  ratio, criterion,
                                                                                                  patience,
                                                                                                  temp_file_name)

    with open("g-test.pkl", "rb") as f1:
        temp = pickle.load(f1)
        test_dataset_temp = {"input_ids": temp['input_ids'],
                             "attention_mask": temp['attention_mask'],
                             "token_type_ids": temp['token_type_ids'],
                             "text_simi_image_feature": temp['text_simi_image_feature'],
                             "image": temp['image'],
                             "label": temp['label'],
                             "similarity": temp['similarity']}
    test_dataset = Rumor_Data(test_dataset_temp)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    test_main(args, test_loader, batch_size, ratio, best_validate_dir, mean_best, std_best, t_c_best, t_d_best,
              temp_file_name, external_memory_bank)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = '../Data/weibo/train.pickle'
    test = '../Data/weibo/test.pickle'
    # output = '../Data/weibo/RESULT/'
    output = 'D:\\glf\\model\\'
    args = parser.parse_args([train, test, output])
    main(args)
