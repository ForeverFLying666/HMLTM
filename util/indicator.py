import torch


def precision_true(pred, target):
    with torch.no_grad():
        count_0, count_correct = 0, 0
        for i in range(len(pred)):
            if pred[i] == 0:
                count_0 += 1
                if pred[i] == target[i]:
                    count_correct += 1
        prec = float(count_correct) / float(count_0)
        return prec


def recall_true(pred, target):
    with torch.no_grad():
        count_correct, count_target_0 = 0, 0
        for i in range(len(pred)):
            if pred[i] == 0 and pred[i] == target[i]:
                count_correct += 1
            if target[i] == 0:
                count_target_0 += 1
        rec = float(count_correct) / float(count_target_0)
        return rec


def f1_true(pred, target):
    with torch.no_grad():
        count_predict_0, count_0_correct, count_target_0 = 0, 0, 0
        for i in range(len(pred)):
            if pred[i] == 0:
                count_predict_0 += 1
            if pred[i] == 0 and pred[i] == target[i]:
                count_0_correct += 1
        count_target_0 = len(list(target))-sum(list(target))
        precision = float(count_0_correct) / float(count_predict_0)
        recall = float(count_0_correct) / float(count_target_0)
        f = 2 * (precision * recall) / (precision + recall)
        return f


def precision_fake(pred, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        count_1,count_correct=0,0
        for i in range(len(pred)):
            if pred[i]==1:
                count_1+=1
                if pred[i]==target[i]:
                    count_correct+=1
        try:
            prec=float(count_correct) / float(count_1)
        except ZeroDivisionError:
            prec=None
        return prec


def recall_fake(pred, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        count_correct,count_target_1=0,0
        for i in range(len(pred)):
            if pred[i]==1 and pred[i]==target[i]:
                count_correct+=1
            if target[i]==1:
                count_target_1+=1
        try:
            rec=float(count_correct) / float(count_target_1)
        except ZeroDivisionError:
            rec=None
        return rec


def f1_fake(pred, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        count_predict_1,count_1_correct,count_target_1=0,0,0
        for i in range(len(pred)):
            if pred[i]==1:
                count_predict_1+=1
            if pred[i]==1 and pred[i]==target[i]:
                count_1_correct+=1
        count_target_1=sum(target)
        try:
            precision=float(count_1_correct)/float(count_predict_1)
        except ZeroDivisionError:
            precision=None
        try:
            recall=float(count_1_correct)/float(count_target_1)
        except ZeroDivisionError:
            recall=None
        if not precision or not recall:
            f=None
        else:
            try:
                f= 2*(precision*recall)/(precision+recall)
            except ZeroDivisionError:
                f = None

        return f
