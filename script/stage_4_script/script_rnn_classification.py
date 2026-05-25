from local_code.stage_4_code.Dataset_Loader_RNN_Classification import Dataset_Loader_RNN_Classification
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Pre_Split import Setting_Pre_Split
from local_code.stage_4_code.Evaluation_Metrics import Evaluate_Accuracy
import numpy as np
import torch

#---- RNN Classification script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization selection ---------------
    data_obj = Dataset_Loader_RNN_Classification('IMDB', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification'

    # load data ahead of time initialize pass vocabulary_size and pad_index before method
    # needed to initialize embedding and output layers
    loaded_data = data_obj.load()

    # SET CELL TYPE HERE
    cell='LSTM'

    method_obj = Method_RNN_Classification(
        'RNN_CLASSIFIER', '',
        vocabulary_size=loaded_data['vocabulary_size'],
        pad_index=loaded_data['pad_index'],
        num_classes=2,
        cell_type=cell
    )

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = f'../../result/stage_4_result/RNN_{method_obj.cell_type}_CLASSIFICATION_'
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
    print(f'RNN ({method_obj.cell_type}) CLASSIFICATION Accuracy: ' + str(metrics['accuracy']))
    print(f'RNN ({method_obj.cell_type}) CLASSIFICATION F1 (macro): ' + str(metrics['f1']))
    print(f'RNN ({method_obj.cell_type}) CLASSIFICATION Precision (macro): ' + str(metrics['precision']))
    print(f'RNN ({method_obj.cell_type}) CLASSIFICATION Recall (macro): ' + str(metrics['recall']))
    print('************ Finish ************')
    # ------------------------------------------------------
