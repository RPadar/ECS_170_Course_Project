from local_code.stage_3_code.Dataset_Loader_CIFAR import Dataset_Loader_CIFAR
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Pre_Split import Setting_Pre_Split
from local_code.stage_3_code.Evaluation_Metrics import Evaluate_Accuracy
import numpy as np
import torch

#---- CNN-CIFAR script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization selection ---------------
    data_obj = Dataset_Loader_CIFAR('CIFAR', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'

    data_obj.dataset_source_file_name = 'CIFAR'


    method_obj = Method_CNN_CIFAR('CNN_CIFAR', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Pre_Split('pre split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN CIFAR Accuracy: ' + str(metrics['accuracy']))
    print('CNN CIFAR F1 (macro): ' + str(metrics['f1']))
    print('CNN CIFAR Precision (macro): ' + str(metrics['precision']))
    print('CNN CIFAR Recall (macro): ' + str(metrics['recall']))
    print('************ Finish ************')
    # ------------------------------------------------------
