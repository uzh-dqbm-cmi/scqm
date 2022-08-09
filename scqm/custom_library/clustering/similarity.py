import torch


# def mse(t1: torch.tensor, t2: list):
#     # if t2 is one dimensionnal
#     if len(t2.shape) == 1:
#         return torch.sum((t1 - t2) ** 2) / len(t1)
#     # if t2 has more than one dimensions, compute mse between t1 and all dimensions (return vector)
#     else:
#         return torch.sum((t1 - t2) ** 2, dim=1) / len(t1)


def mse(t1: torch.tensor, t2: torch.tensor):
    return torch.sum((t1 - t2) ** 2, dim=1) / t1.shape[1]


def cosine(t1: torch.tensor, t2: torch.tensor):

    return torch.nn.CosineSimilarity(dim=1)(t1, t2)


def compute_similarity(
    patient_id: str,
    other_ids: list,
    representations: torch.tensor,
    patient_in_embedding: dict,
    name: str = "mse",
):
    other_representations = [
        representations[patient_in_embedding[p]["indices"]] for p in other_ids
    ]
    similarities = {}
    rep = representations[patient_in_embedding[patient_id]["indices"][-1]].reshape(
        (1, -1)
    )
    if name == "mse":
        for index, p in enumerate(other_representations):
            similarities[other_ids[index]] = mse(rep, p)

    elif name == "cosine":
        for index, p in enumerate(other_representations):
            similarities[other_ids[index]] = cosine(rep, p)
    return similarities
