from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Pre_Split import Setting_Pre_Split
from local_code.stage_2_code.Evaluation_Metrics import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization selection ---------------
    data_obj = Dataset_Loader('stage 2', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'

    data_obj.dataset_source_file_name_train = 'train.csv'
    data_obj.dataset_source_file_name_test = 'test.csv'


    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Pre_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(metrics['accuracy']))
    print('MLP F1 (macro): ' + str(metrics['f1']))
    print('MLP Precision (macro): ' + str(metrics['precision']))
    print('MLP Recall (macro): ' + str(metrics['recall']))
    print('************ Finish ************')
    # ------------------------------------------------------
