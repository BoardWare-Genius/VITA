import warnings

from .image_base import ImageBaseDataset
from .image_mcq import ImageMCQDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *

class TourMCQ(ImageMCQDataset):
    TYPE = 'MCQ'

    DATASET_URL = {'TOUR-MO': '/Dataset/Domain/LMUData/test/TourMO_MCQ.json',
                   'YUE-MMLU' : '/Dataset/Domain/LMUData/yue_benchmark_v1/Yue-MMLU'
                   }

    DATASET_MD5 = {}

    # def evaluate(self, eval_file, **judge_kwargs):
    #     from .utils.multiple_choice import report_acc, report_acc_MMT, mcq_circular_eval, mcq_vanilla_eval

    def load_data(self, dataset):
        data_path = self.DATASET_URL[dataset]
        if dataset == 'YUE-MMLU':
            data_list = []
            for file_name in os.listdir(data_path):
                dat = pd.DataFrame(load(osp.join(data_path,file_name)))
                data_list.append(dat)
            data = pd.concat(data_list)
            data['index'] = data.index
        else:
            data = pd.DataFrame(load(data_path))
            data['index'] = data['no']      
        return data