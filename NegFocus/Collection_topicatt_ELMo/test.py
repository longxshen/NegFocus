#!/bin/python
# encoding: utf-8
from train import *

# main function
model = torch.load('models/saved_model_70.08')
best_test_Acc, new_test_Acc, best_test_Acc_post, new_test_Acc_post, _ = evaluating(model, test_data, best_test_Acc, best_test_Acc_post, True, 1)
print('test Acc: ' + str(new_test_Acc))
print('test Acc postprocessing: ' + str(new_test_Acc_post))


