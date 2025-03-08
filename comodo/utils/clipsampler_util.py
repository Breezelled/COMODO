from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from typing import Dict, Any

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomHorizontalFlip,
    Resize,
)

class DynamicClipsPerVideoSampler(ClipSampler):
    """
    Splits all possible segments based on the total video duration and clip_duration.
    """

    def __init__(self, clip_duration: float, overlap: float = 0.0) -> None:
        """
        Initializes the sampler with the specified clip duration and overlap.

        Args:
            clip_duration (float): The duration (in seconds) for each clip.
            overlap (float): The fraction of overlap between consecutive clips (default: 0.0).
        """
        super().__init__(clip_duration)
        self._overlap = overlap

    def get_num_clips(self, video_duration: float) -> int:
        stride = self._clip_duration * (1 - self._overlap)
        return max(1, int(video_duration // stride))

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:
        """
        Generates clip information based on the video length and controls overlap.

        Args:
            last_clip_time (float): Time of the last clip (not used in this sampler).
            video_duration (float): The total duration of the video in seconds.
            annotation (Dict): Annotations for the video (not used here).

        Returns:
            ClipInfo: Contains information about the current clip, including
            (clip_start_time, clip_end_time, clip_index, aug_index, is_last_clip).
        """
        # Calculate the stride between consecutive clips, adjusted for overlap
        stride = self._clip_duration * (1 - self._overlap)
        clip_start_sec = stride * self._current_clip_index

        # Predict if the next clip would exceed video duration
        next_clip_start_sec = clip_start_sec + stride
        is_last_clip = next_clip_start_sec + self._clip_duration > video_duration

        # Create a ClipInfo object with the clip's start and end times
        clip_info = ClipInfo(
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_start_sec + self._clip_duration,
            clip_index=self._current_clip_index,
            aug_index=0,
            is_last_clip=is_last_clip,
        )

        # Update the state to prepare for the next clip
        self._current_clip_index += 1

        # Reset the index if this was the last clip
        if is_last_clip:
            self.reset()

        return clip_info

    def reset(self):
        """Resets the clip and augmentation indices."""
        self._current_clip_index = 0


def get_uestc_video_transform_clipsampler(video_teacher, sample_rate, fps):
    num_frames_to_sample = video_teacher.model.config.num_frames
    clip_duration = num_frames_to_sample * sample_rate / fps

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(
                            video_teacher.image_mean,
                            video_teacher.image_std,
                        ),
                        Resize((224,  224), antialias=True),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(
                            video_teacher.image_mean,
                            video_teacher.image_std,
                        ),
                        Resize((224, 224), antialias=True),
                    ]
                ),
            ),
        ]
    )

    return train_transform, val_transform, None, None

def get_ego4d_video_transform_clipsampler(video_teacher, sample_rate, fps):
    num_frames_to_sample = video_teacher.model.config.num_frames
    clip_duration = num_frames_to_sample * sample_rate / fps
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(
                            video_teacher.image_mean, video_teacher.image_std
                        ),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(
                            video_teacher.image_mean, video_teacher.image_std
                        ),
                    ]
                ),
            ),
        ]
    )

    clip_sampler = DynamicClipsPerVideoSampler(clip_duration, overlap=0);

    return train_transform, val_transform, clip_sampler, clip_sampler