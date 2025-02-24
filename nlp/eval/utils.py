import copy
import torch
import os


def prepare_batchOfEvalInfo(batch):
    batchOf_evalInfo = copy.deepcopy(batch)

    for (key, value) in batch.items():
        # Remove ids and mask since no longer needed
        if ("ids" in key) or ("mask" in key):
            del batchOf_evalInfo[key]
        else:
            # Convert tensors to list
            if torch.is_tensor(batchOf_evalInfo[key]):
                batchOf_evalInfo[key] = value.cpu().numpy().tolist()

    return batchOf_evalInfo