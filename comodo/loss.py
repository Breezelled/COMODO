import torch
import torch.nn as nn
import torch.nn.functional as F


class COMODOLoss(nn.Module):
    def __init__(
        self,
        instanceQ_encoded,
        student_model,
        teacher_temp=0.1,
        student_temp=0.05,
    ):
        """
        student_model:    IMU model
        teacher_model:    Video model
        teacher_temp:   distillation temperature for teacher model
        student_temp:   distillation temperature for student model
        """
        super(COMODOLoss, self).__init__()
        self.instanceQ_encoded = instanceQ_encoded
        self.student_model = student_model
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(
        self,
        imu_features: torch.Tensor,
        z_v: torch.Tensor,
        input_mask: torch.Tensor = None,
    ):

        batch_size = z_v.shape[0]

        z_x = F.normalize(self.student_model(imu_features, input_mask), p=2, dim=1)

        # insert the current batch embedding from T
        instanceQ_encoded = self.instanceQ_encoded
        Q = torch.cat((instanceQ_encoded, z_v))

        # probability scores distribution for T, S: B X (N + 1)
        P_v = torch.einsum("nc,ck->nk", z_v, Q.t().clone().detach())
        P_x = torch.einsum("nc,ck->nk", z_x, Q.t().clone().detach())

        # FKL
        # Apply temperatures for soft-labels
        P_v = F.softmax(P_v / self.teacher_temp, dim=1)
        P_x = P_x / self.student_temp

        # loss computation, use log_softmax for stable computation
        loss = -torch.mul(P_v, F.log_softmax(P_x, dim=1)).sum() / batch_size

        # update the random sample queue
        self.instanceQ_encoded = Q[batch_size:]

        return loss


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(
        self,
        temperature=0.1,
        reduction="mean",
        negative_mode="unpaired",
        symmetric_loss=False,
        learn_temperature=False,
    ):
        super().__init__()
        self.temperature = (
            nn.Parameter(torch.tensor(temperature))
            if learn_temperature
            else temperature
        )
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.symmetric_loss = symmetric_loss

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(
            query,
            positive_key,
            negative_keys,
            temperature=self.temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode,
            symmetric_loss=self.symmetric_loss,
        )


def info_nce(
    query,
    positive_key,
    negative_keys=None,
    temperature=0.1,
    reduction="mean",
    negative_mode="unpaired",
    symmetric_loss=False,
):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError("<query> must have 2 dimensions.")
    if positive_key.dim() != 2:
        raise ValueError("<positive_key> must have 2 dimensions.")
    if negative_keys is not None:
        if negative_mode == "unpaired" and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'."
            )
        if negative_mode == "paired" and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'."
            )

    # Check matching number of samples.
    if len(query) != len(positive_key):
        print("Query shape:", query.shape)
        print("Positive key shape:", positive_key.shape)
        raise ValueError(
            "<query> and <positive_key> must must have the same number of samples."
        )
    if negative_keys is not None:
        if negative_mode == "paired" and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>."
            )

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            "Vectors of <query> and <positive_key> should have the same number of components."
        )
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                "Vectors of <query> and <negative_keys> should have the same number of components."
            )

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == "unpaired":
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == "paired":
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    if symmetric_loss:
        # TODO: consider use learned temperature
        loss_i = F.nll_loss(F.log_softmax(logits / temperature, dim=0), labels)
        loss_t = F.nll_loss(F.log_softmax(logits / temperature, dim=1), labels)
        return loss_i + loss_t
    else:
        return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class L2DistillLoss(nn.Module):
    def __init__(self):
        super(L2DistillLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor):
        student_features = F.normalize(student_features, p=2, dim=1)

        loss = self.mse_loss(student_features, teacher_features)
        return loss
