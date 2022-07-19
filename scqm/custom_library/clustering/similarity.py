import torch


def mse(t1: torch.tensor, t2: torch.tensor):
    # if t2 is one dimensionnal
    if len(t2.shape) == 1:
        return torch.sum((t1 - t2) ** 2) / len(t1)
    # if t2 has more than one dimensions, compute mse between t1 and all dimensions (return vector)
    else:
        return torch.sum((t1 - t2) ** 2, dim=1) / len(t1)


def cosine(t1: torch.tensor, t2: torch.tensor):
    if len(t2.shape) == 1:
        return torch.nn.CosineSimilarity(dim=0)(t1, t2)
    else:
        return torch.nn.CosineSimilarity(dim=1)(t1, t2)


def compute_similarity(
    patient_id: str,
    other_ids: list,
    representations: torch.tensor,
    indices_mapping: dict,
    name: str = "mse",
):
    other_representations = representations[[indices_mapping[p] for p in other_ids]]
    if name == "mse":
        similarities = mse(
            representations[indices_mapping[patient_id]], other_representations
        )
    elif name == "cosine":
        similarities = cosine(
            representations[indices_mapping[patient_id]], other_representations
        )
    return similarities
