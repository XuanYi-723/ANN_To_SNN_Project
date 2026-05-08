import torch

from models.ann_model import NetANN
from models.snn_model import NetSNN

def load_ann_model(model_path):

    ann_model = NetANN()

    ann_model.load_state_dict(
        torch.load(model_path)
    )

    ann_model.eval()

    return ann_model


def convert_to_snn(model_path):

    ann_model = load_ann_model(model_path)

    snn_model = NetSNN(ann_model)

    return ann_model, snn_model
