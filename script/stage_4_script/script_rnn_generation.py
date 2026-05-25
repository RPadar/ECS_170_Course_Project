from local_code.stage_4_code.Dataset_Loader_RNN_Generation import Dataset_Loader_RNN_Generation
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
import numpy as np
import torch

#---- RNN Classification script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization selection ---------------
    data_obj = Dataset_Loader_RNN_Generation('jokes', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'

    # load data ahead of time initialize pass vocabulary_size and pad_index before method
    # needed to initialize embedding and output layers
    loaded_data = data_obj.load()

    # SET CELL TYPE HERE
    cell='GRU'

    method_obj = Method_RNN_Generation(
        'RNN_GENERATOR', '',
        vocabulary_size=loaded_data['vocabulary_size'],
        pad_index=loaded_data['pad_index'],
        cell_type=cell
    )
    # pass vocabularies to method
    method_obj.word_to_index = data_obj.word_to_index
    method_obj.index_to_word = data_obj.index_to_word
    method_obj.data = loaded_data
    # ensure method and dataset window sizes align
    method_obj.window_size = data_obj.window_size

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    method_obj.run()
    print('************ Finish ************')
    # ------------------------------------------------------
