from momentfm import MOMENTPipeline
from chronos import ChronosBoltPipeline
from mantis.architecture import Mantis8M

from transformers import (
    VideoMAEModel,
    VideoMAEImageProcessor,
    AutoImageProcessor,
    TimesformerModel,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def MLP(input_dim, output_dim, hidden_dim, activation_fn=nn.GELU):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        activation_fn,
        nn.Linear(hidden_dim, output_dim),
    )


def resize(X):
    X_scaled = F.interpolate(
        torch.tensor(X, dtype=torch.float).to(device),
        size=512,
        mode="linear",
        align_corners=False,
    )
    return X_scaled


class FineTuningNetwork(nn.Module):
    """
    A nn.Module wrapper to combine adapter, encoder and prediction head.

    Parameters
    ----------
    encoder: nn.Module
        The encoder (foundation model) that projects from ``(n_samples, n_channels, seq_len)`` to
        ``(n_samples, hidden_dim)``. If None, it is assumed that the input matrix represents already the embeddings, so
        the input is directly passed through ``head``.
    head: nn.Module, default=None
        Head is a part of the network that follows the foundation model and projects from the embedding space of shape
        ``(n_samples, hidden_dim)`` to the probability matrix of shape ``(n_samples, n_classes)``. The way this class
        is implemented, ``head`` cannot be None.
    adapter: nn.Module, default=None
        Adapter is a part of the network that precedes the foundation model and reduces the original data matrix
        of shape ``(n_samples, n_channels, seq_len)`` to ``(n_samples, new_n_channels, seq_len)``.
        By default, adapter is None, i.e., not used.
    """

    def __init__(self, encoder, head, adapter=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.adapter = adapter

    def forward(self, x):
        if self.encoder is None:
            return self.head(x)
        elif self.adapter is None:
            if x.shape[1] > 1:
                Warning(
                    "The data is multi-variate! Applying encoder to all channels independently"
                )
                return self.head(
                    torch.cat(
                        [self.encoder(x[:, [i], :]) for i in range(x.shape[1])], dim=-1
                    )
                )
            else:
                return self.head(self.encoder(x))
        else:
            adapter_output = self.adapter(x)
            return self.head(
                torch.cat(
                    [
                        self.encoder(adapter_output[:, [i], :])
                        for i in range(adapter_output.shape[1])
                    ],
                    dim=-1,
                )
            )


class VideoTeacher(nn.Module):
    def __init__(self, model_name, device, **kwargs):
        super(VideoTeacher, self).__init__()
        if "videomae" in model_name.lower():
            self.model = VideoMAEModel.from_pretrained(model_name, **kwargs)
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
            self.use_mean_pooling = self.model.config.use_mean_pooling
        elif "timesformer" in model_name.lower():
            self.model = TimesformerModel.from_pretrained(model_name, **kwargs)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.use_mean_pooling = True
        self.device = device
        self.to(self.device)
        self.model.eval()

    @property
    def image_mean(self):
        return self.processor.image_mean

    @property
    def image_std(self):
        return self.processor.image_std

    @property
    def size(self):
        return self.processor.size

    def encode(self, video_tensor: torch.Tensor, normalize_embeddings: bool = True):

        with torch.no_grad():
            # video_tensor: [batch_size, num_frames, num_channels, height, width]
            video_tensor = video_tensor.to(self.device)

            # video_hidden_state: [batch_size, num_frames, hidden_size]
            video_hidden_state, *_ = self.model(video_tensor, return_dict=False)

            # video_embeddings: [batch_size, hidden_size]
            if self.use_mean_pooling:
                video_embeddings = video_hidden_state.mean(dim=1)
            else:
                video_embeddings = video_hidden_state[:, 0]

            if normalize_embeddings:
                video_embeddings = F.normalize(video_embeddings, p=2, dim=1)

        return video_embeddings


class VideoTeacherMLP(nn.Module):
    def __init__(
        self,
        model_name,
        mlp_output_dim: int,
        mlp_hidden_dim: int,
        device,
        activation_fn=nn.GELU,
        **kwargs,
    ):
        super(VideoTeacherMLP, self).__init__()
        if "videomae" in model_name.lower():
            self.model = VideoMAEModel.from_pretrained(model_name, **kwargs)
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
            self.use_mean_pooling = self.model.config.use_mean_pooling
        elif "timesformer" in model_name.lower():
            self.model = TimesformerModel.from_pretrained(model_name, **kwargs)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.use_mean_pooling = True
        self.device = device
        self.activation_fn = activation_fn()
        self.mlp = MLP(
            self.model.config.hidden_size,
            mlp_output_dim,
            mlp_hidden_dim,
            self.activation_fn,
        )
        self.to(self.device)
        self.eval()

    @property
    def image_mean(self):
        return self.processor.image_mean

    @property
    def image_std(self):
        return self.processor.image_std

    @property
    def size(self):
        return self.processor.size

    def encode(self, video_tensor: torch.Tensor, normalize_embeddings: bool = True):

        with torch.no_grad():
            # video_tensor: [batch_size, num_frames, num_channels, height, width]
            video_tensor = video_tensor.to(self.device)

            # video_hidden_state: [batch_size, num_frames, hidden_size]
            video_hidden_state, *_ = self.model(video_tensor, return_dict=False)

            # video_embeddings: [batch_size, hidden_size]
            if self.use_mean_pooling:
                video_embeddings = video_hidden_state.mean(dim=1)
            else:
                video_embeddings = video_hidden_state[:, 0]

            video_embeddings = self.mlp(video_embeddings)

            if normalize_embeddings:
                video_embeddings = F.normalize(video_embeddings, p=2, dim=1)

            del video_tensor, video_hidden_state
        return video_embeddings


class IMUStudent(nn.Module):
    def __init__(
        self,
        imu_student: MOMENTPipeline | ChronosBoltPipeline | Mantis8M,
        device,
        teacher_dimension: int,
        activation_fn=nn.GELU,
        reduction="concat",
    ):
        super(IMUStudent, self).__init__()
        self.imu_student = imu_student
        if isinstance(self.imu_student, MOMENTPipeline):
            self.student_dimension = self.imu_student.config.d_model * 6
        elif isinstance(self.imu_student, Mantis8M):
            self.student_dimension = self.imu_student.hidden_dim * 6
        elif isinstance(self.imu_student, ChronosBoltPipeline):
            self.student_dimension = self.imu_student.model.config.d_model * 6
        else:
            raise NotImplementedError("Unsupported imu_student type")
        self.teacher_dimension = teacher_dimension
        self.activation_fn = activation_fn() if activation_fn is not None else None
        self.reduction = reduction
        self.dense = nn.Linear(self.student_dimension, self.teacher_dimension)
        self.use_projection = True
        self.to(device)

    def forward(self, sensor_values: torch.Tensor, input_mask: torch.Tensor = None):
        if isinstance(self.imu_student, MOMENTPipeline):
            outputs = self.imu_student(
                x_enc=sensor_values, reduction=self.reduction, input_mask=input_mask
            )
            # imu_embeddings: [batch_size, n_patches, hidden_size]
            imu_embeddings = outputs.embeddings

            # imu_embeddings: [batch_size, hidden_size]
            imu_embeddings = imu_embeddings.mean(dim=1)
        elif isinstance(self.imu_student, Mantis8M):
            batch_size, n_channels, seq_len = sensor_values.shape
            if seq_len != 512:
                sensor_values = resize(sensor_values)

            channel_embeddings = []

            for ch in range(n_channels):
                # Extract data for current channel
                channel_data = sensor_values[:, ch : ch + 1, :]

                # Forward pass through tokgen_unit to get patch embeddings
                channel_embedding = self.imu_student(channel_data)

                channel_embeddings.append(channel_embedding)

            # Concatenate all channel embeddings
            imu_embeddings = torch.cat(channel_embeddings, dim=1)
        elif isinstance(self.imu_student, ChronosBoltPipeline):
            # sensor_values: [batch_size, n_channel, seq_len]
            batch_size, n_channel, seq_len = sensor_values.shape
            channel_embeddings = []

            for ch in range(n_channel):
                # Extract data for the current channel, shape: [batch_size, seq_len]
                context_ch = sensor_values[:, ch, :]
                mask_ch = input_mask[:, ch, :] if input_mask is not None else None

                # Use the ChronosBolt model's encode method.
                # Note: self.imu_student.model.encode returns a tuple:
                # (encoder_output, loc_scale, input_embeds, attention_mask)
                encoder_output, *_ = self.imu_student.model.encode(
                    context=context_ch, mask=mask_ch
                )
                # encoder_output: [batch_size, patched_seq_length, d_model]
                # Aggregate over the patched sequence dimension.
                # Here, we use mean pooling to get a single embedding per channel.
                ch_embedding = encoder_output.mean(
                    dim=1
                )  # shape: [batch_size, d_model]
                channel_embeddings.append(ch_embedding)

            # Concatenate embeddings from all channels along the feature dimension.
            # Final imu_embeddings shape: [batch_size, n_channel * d_model]
            imu_embeddings = torch.cat(channel_embeddings, dim=1)
        else:
            raise NotImplementedError("Unsupported imu_student type")

        if self.use_projection:
            # imu_embeddings: [batch_size, teacher_dimension]
            if self.activation_fn is not None:
                imu_embeddings = self.activation_fn(self.dense(imu_embeddings))
            else:
                imu_embeddings = self.dense(imu_embeddings)
        return imu_embeddings

    def remove_projection_layer(self):
        self.use_projection = False


class IMUStudentMLP(nn.Module):
    def __init__(
        self,
        imu_student: (
            MOMENTPipeline | ChronosBoltPipeline | Mantis8M | FineTuningNetwork
        ),
        device,
        mlp_output_dim: int,
        mlp_hidden_dim: int,
        activation_fn=nn.GELU,
        reduction="concat",
    ):
        super(IMUStudentMLP, self).__init__()
        self.imu_student = imu_student
        if isinstance(self.imu_student, MOMENTPipeline):
            if reduction == "mean":
                self.student_dimension = self.imu_student.config.d_model
            elif reduction == "concat":
                self.student_dimension = self.imu_student.config.d_model * 6
        elif isinstance(self.imu_student, Mantis8M):
            self.student_dimension = self.imu_student.hidden_dim * 6
        elif isinstance(self.imu_student, FineTuningNetwork):
            self.student_dimension = self.imu_student.encoder.hidden_dim * 6
        elif isinstance(self.imu_student, ChronosBoltPipeline):
            self.student_dimension = self.imu_student.model.config.d_model * 6
        self.mlp_output_dim = mlp_output_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.activation_fn = activation_fn()
        self.reduction = reduction
        self.use_projection = True
        # self.dense = nn.Linear(self.student_dimension, self.teacher_dimension)
        self.mlp = MLP(
            self.student_dimension,
            self.mlp_output_dim,
            self.mlp_hidden_dim,
            self.activation_fn,
        )
        self.to(device)

    def forward(self, sensor_values: torch.Tensor, input_mask: torch.Tensor = None):
        if isinstance(self.imu_student, MOMENTPipeline):
            outputs = self.imu_student(
                x_enc=sensor_values, reduction=self.reduction, input_mask=input_mask
            )
            # imu_embeddings: [batch_size, n_patches, hidden_size]
            imu_embeddings = outputs.embeddings

            # imu_embeddings: [batch_size, hidden_size]
            imu_embeddings = imu_embeddings.mean(dim=1)
        elif isinstance(self.imu_student, Mantis8M):
            batch_size, n_channels, seq_len = sensor_values.shape
            if seq_len != 512:
                sensor_values = resize(sensor_values)

            channel_embeddings = []

            for ch in range(n_channels):
                # Extract data for current channel
                channel_data = sensor_values[:, ch : ch + 1, :]

                # Forward pass through tokgen_unit to get patch embeddings
                channel_embedding = self.imu_student(channel_data)

                channel_embeddings.append(channel_embedding)

            # Concatenate all channel embeddings
            imu_embeddings = torch.cat(channel_embeddings, dim=1)
        elif isinstance(self.imu_student, FineTuningNetwork):
            batch_size, n_channels, seq_len = sensor_values.shape
            if seq_len != 512:
                sensor_values = resize(sensor_values)

            channel_embeddings = []

            for ch in range(n_channels):
                # Extract data for current channel
                channel_data = sensor_values[:, ch : ch + 1, :]

                # Forward pass through tokgen_unit to get patch embeddings
                channel_embedding = self.imu_student.encoder(channel_data)

                channel_embeddings.append(channel_embedding)

            # Concatenate all channel embeddings
            imu_embeddings = torch.cat(channel_embeddings, dim=1)
        elif isinstance(self.imu_student, ChronosBoltPipeline):
            # sensor_values: [batch_size, n_channel, seq_len]
            batch_size, n_channel, seq_len = sensor_values.shape
            channel_embeddings = []

            for ch in range(n_channel):
                # Extract data for the current channel, shape: [batch_size, seq_len]
                context_ch = sensor_values[:, ch, :]
                mask_ch = input_mask[:, ch, :] if input_mask is not None else None

                # Use the ChronosBolt model's encode method.
                # Note: self.imu_student.model.encode returns a tuple:
                # (encoder_output, loc_scale, input_embeds, attention_mask)
                encoder_output, *_ = self.imu_student.model.encode(
                    context=context_ch, mask=mask_ch
                )
                # encoder_output: [batch_size, patched_seq_length, d_model]
                # Aggregate over the patched sequence dimension.
                # Here, we use mean pooling to get a single embedding per channel.
                ch_embedding = encoder_output.mean(
                    dim=1
                )  # shape: [batch_size, d_model]
                channel_embeddings.append(ch_embedding)

            # Concatenate embeddings from all channels along the feature dimension.
            # Final imu_embeddings shape: [batch_size, n_channel * d_model]
            imu_embeddings = torch.cat(channel_embeddings, dim=1)
            # imu_embeddings = torch.stack(channel_embeddings, dim=1)
            # imu_embeddings = imu_embeddings.mean(dim=1)
        else:
            raise NotImplementedError("Unsupported imu_student type")

        # imu_embeddings: [batch_size, teacher_dimension]
        # imu_embeddings = self.activation_fn(self.dense(imu_embeddings))
        if self.use_projection:
            imu_embeddings = self.mlp(imu_embeddings)

        return imu_embeddings

    def remove_projection_layer(self):
        self.use_projection = False


def imu_student(teacher_dimension, imu_ckpt, num_classes, activation_fn, n_channels=6):
    if "moment" in imu_ckpt.lower():
        imu_student = MOMENTPipeline.from_pretrained(
            imu_ckpt,
            model_kwargs={
                "task_name": "classification",
                "n_channels": n_channels,
                "num_class": num_classes,
                "freeze_encoder": False,
                "freeze_embedder": False,
                "reduction": "concat",
            },
        )
        imu_student.to(device)
        imu_student.init()
    elif "mantis" in imu_ckpt.lower():
        imu_student = Mantis8M(device=device)
        imu_student = imu_student.from_pretrained(imu_ckpt)
    imu_student = IMUStudent(
        imu_student, device, teacher_dimension, activation_fn, reduction="concat"
    )
    imu_student.to(device)
    return imu_student


def imu_student_mlp(
    mlp_output_dim,
    imu_ckpt,
    num_classes,
    mlp_hidden_dim=2048,
    model_path=None,
    n_channels=6,
    freeze=False,
):
    if "chronos" in imu_ckpt.lower():
        imu_student = ChronosBoltPipeline.from_pretrained(
            imu_ckpt,
        )
        imu_student.model.to(device)
    elif "moment" in imu_ckpt.lower():
        imu_student = MOMENTPipeline.from_pretrained(
            imu_ckpt,
            model_kwargs={
                "task_name": "classification",
                "n_channels": n_channels,
                "num_class": num_classes,
                "freeze_encoder": freeze,
                "freeze_embedder": freeze,
                "reduction": "concat",
            },
        )
        imu_student.to(device)
        imu_student.init()
    elif "mantis" in imu_ckpt.lower():
        imu_student = Mantis8M(device=device)
        imu_student = imu_student.from_pretrained(imu_ckpt)

        if model_path is not None:
            head = nn.Sequential(
                nn.LayerNorm(imu_student.hidden_dim * n_channels),
                nn.Linear(imu_student.hidden_dim * n_channels, num_classes),
            ).to(device)
            imu_student = FineTuningNetwork(imu_student, head, adapter=None).to(device)
            imu_student.load_state_dict(torch.load(model_path, map_location=device))

    imu_student = IMUStudentMLP(
        imu_student,
        device,
        mlp_output_dim,
        mlp_hidden_dim,
        activation_fn=nn.GELU,
        reduction="concat",
    )
    imu_student.to(device)
    return imu_student


def create_pipeline(
    model_name, num_classes, device, reduction, n_channels=6
) -> ChronosBoltPipeline | MOMENTPipeline:
    if model_name.startswith("AutonLab"):
        imu_student = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={
                "task_name": "classification",
                "n_channels": n_channels,
                "num_class": num_classes,
                "freeze_encoder": False,
                "freeze_embedder": False,
                "reduction": reduction,
            },
        )
        imu_student.to(device)
        imu_student.init()
        return imu_student

    elif model_name.startswith("amazon"):
        imu_student = ChronosBoltPipeline.from_pretrained(
            model_name,
        )
        imu_student.model.to(device)
        return imu_student
    elif model_name.startswith("paris"):
        imu_student = Mantis8M(device=device)
        imu_student = imu_student.from_pretrained(model_name)
        return imu_student
