from local_code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader_ORL
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Pre_Split import Setting_Pre_Split
from local_code.stage_3_code.Evaluation_Metrics import Evaluate_Accuracy
import numpy as np
import torch

#---- CNN-ORL script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization selection ---------------
    data_obj = Dataset_Loader_ORL('ORL', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'

    data_obj.dataset_source_file_name = 'ORL'


    method_obj = Method_CNN_ORL('CNN_ORL', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_ORL_'
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
    print('CNN ORL Accuracy: ' + str(metrics['accuracy']))
    print('CNN ORL F1 (macro): ' + str(metrics['f1']))
    print('CNN ORL Precision (macro): ' + str(metrics['precision']))
    print('CNN ORL Recall (macro): ' + str(metrics['recall']))
    print('************ Finish ************')
    # ------------------------------------------------------
