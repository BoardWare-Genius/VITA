import json
import pandas as pd
import torch.utils.data.sampler

with open('/Dataset/Domain/LMUData/yue_benchmark_v1/Yue-ARC-C/Yue-ARC-C.json','r', encoding='utf-8') as f:
    data = json.load(f)

    import pdb; pdb.set_trace()
    dat  = pd.DataFrame(data)
    dat['index'] = dat['no']
     