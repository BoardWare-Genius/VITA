import warnings

from .image_base import ImageBaseDataset
from .image_yorn import ImageYORNDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *

class TourYN(ImageYORNDataset):
    TYPE = 'Y/N'

    DATASET_URL = {
        # Lanchao
        'TOUR-YN': '/iivanwu/data/LMUData/test/TourMO_Entity_YN.json',
        # Zstack
        # 'TOUR-YN': '/Dataset/Domain/LMUData/test/TourMO_Entity_YN.json',
    }

    DATASET_MD5 = {

    }

    def load_data(self, dataset):
        data_path = self.DATASET_URL[dataset]
        tsv_file = data_path.replace('.json','.tsv')
        data = pd.DataFrame(load(data_path))
        path_prefix = osp.dirname(data_path)
        data['image_path'] = data['image_path'].apply(lambda x: x.replace('./',path_prefix+'/'))
        data.to_csv(tsv_file, sep='\t',index=False)
        update_flag = False
        if file_size(data_path, 'GB') > 1:
            local_path = tsv_file.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from vlmeval.tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return data
