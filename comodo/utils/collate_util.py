import torch
import numpy as np
import torch.nn.functional as F

def UESTC_MMEA_CL_MOMENT_train_collate_fn(examples):
    video_embeddings = torch.tensor(
        np.array([example["encoded_video"] for example in examples])
    )
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]

    max_seq_len = max([imu.shape[1] for imu in imu_values])
    target_length = 512
    
    final_length = max(max_seq_len, target_length)
    
    padded_imu_values = []
    
    input_mask = torch.zeros(len(imu_values), final_length)
    
    for i, imu in enumerate(imu_values):
        orig_len = imu.shape[1]
        
        padding = torch.zeros(imu.shape[0], final_length - orig_len)
        padded_imu = torch.cat([padding, imu], dim=1)
        padded_imu_values.append(padded_imu)
        
        input_mask[i, -orig_len:] = 1
    
    imu_values = padded_imu_values

    # [batch_size, num_channels, seq_len]
    sensor_tensor = torch.stack(imu_values)

    return {
        "imu": sensor_tensor,
        "input_mask": input_mask,
        "encoded_video": video_embeddings,
    }

def UESTC_MMEA_CL_Mantis_train_collate_fn(examples):
    video_embeddings = torch.tensor(
        np.array([example["encoded_video"] for example in examples])
    )
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]
    
    target_length = 512
    resized_imu_values = []
    
    for imu in imu_values:
        imu_expanded = imu.unsqueeze(0)  # [1, num_channels, seq_len]
        imu_resized = F.interpolate(imu_expanded, size=target_length, mode='linear', align_corners=False)
        resized_imu_values.append(imu_resized.squeeze(0))
    
    sensor_tensor = torch.stack(resized_imu_values)  # [batch_size, num_channels, target_length]

    return {
        "imu": sensor_tensor,
        "encoded_video": video_embeddings,
    }

def train_collate_fn(examples):
    video_embeddings = torch.tensor(
        np.array([example["encoded_video"] for example in examples])
    )
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]

    # [batch_size, num_channels, seq_len]
    sensor_tensor = torch.stack(imu_values)

    return {
        "imu": sensor_tensor,
        "encoded_video": video_embeddings,
    }

def eval_collate_fn(examples):
    labels = torch.tensor(
        [example["label"] for example in examples], dtype=torch.long
    )

    # [batch_size, num_channels, seq_len]
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]

    sensor_tensor = torch.stack(imu_values)

    return {"imu": sensor_tensor, "label": labels}

def UESTC_MMEA_CL_MOMENT_eval_collate_fn(examples):
    labels = torch.tensor(
        [example["label"] for example in examples], dtype=torch.long
    )
    # [batch_size, num_channels, seq_len]
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]

    max_seq_len = max([imu.shape[1] for imu in imu_values])
    target_length = 512
    
    final_length = max(max_seq_len, target_length)
    
    padded_imu_values = []
    
    input_mask = torch.zeros(len(imu_values), final_length)
    
    for i, imu in enumerate(imu_values):
        orig_len = imu.shape[1]
        
        padding = torch.zeros(imu.shape[0], final_length - orig_len)
        padded_imu = torch.cat([padding, imu], dim=1)
        padded_imu_values.append(padded_imu)
        
        input_mask[i, -orig_len:] = 1
    
    imu_values = padded_imu_values

    # [batch_size, num_channels, seq_len]
    sensor_tensor = torch.stack(imu_values)

    return {"imu": sensor_tensor, "label": labels, "input_mask": input_mask}

def UESTC_MMEA_CL_Mantis_eval_collate_fn(examples):
    labels = torch.tensor(
        [example["label"] for example in examples], dtype=torch.long
    )

    # [batch_size, num_channels, seq_len]
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]
    
    target_length = 512
    resized_imu_values = []
    
    for imu in imu_values:
        imu_expanded = imu.unsqueeze(0)  # [1, num_channels, seq_len]
        imu_resized = F.interpolate(imu_expanded, size=target_length, mode='linear', align_corners=False)
        resized_imu_values.append(imu_resized.squeeze(0))
    
    sensor_tensor = torch.stack(resized_imu_values)  # [batch_size, num_channels, target_length]

    return {"imu": sensor_tensor, "label": labels}


def UESTC_MMEA_CL_MOMENT_encode_train_collate_fn(examples):
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    # [batch_size, num_channels, seq_len]
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]

    max_seq_len = max([imu.shape[1] for imu in imu_values])
    target_length = 512
    
    final_length = max(max_seq_len, target_length)
    
    padded_imu_values = []
    
    input_mask = torch.zeros(len(imu_values), final_length)
    
    for i, imu in enumerate(imu_values):
        orig_len = imu.shape[1]
        
        padding = torch.zeros(imu.shape[0], final_length - orig_len)
        padded_imu = torch.cat([padding, imu], dim=1)
        padded_imu_values.append(padded_imu)
        
        input_mask[i, -orig_len:] = 1
    
    imu_values = padded_imu_values

    # [batch_size, num_channels, seq_len]
    sensor_tensor = torch.stack(imu_values)

    return {
        "imu": sensor_tensor,
        "input_mask": input_mask,
        "video": pixel_values,
    }

def UESTC_MMEA_CL_Mantis_encode_train_collate_fn(examples):
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]
    
    target_length = 512
    resized_imu_values = []
    
    for imu in imu_values:
        imu_expanded = imu.unsqueeze(0)  # [1, num_channels, seq_len]
        imu_resized = F.interpolate(imu_expanded, size=target_length, mode='linear', align_corners=False)
        resized_imu_values.append(imu_resized.squeeze(0))
    
    sensor_tensor = torch.stack(resized_imu_values)  # [batch_size, num_channels, target_length]
    
    return {
        "imu": sensor_tensor,
        "video": pixel_values,
    }


def EGO_encode_train_collate_fn(examples):
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    imu_values = [torch.tensor(example["imu"], dtype=torch.float) for example in examples]

    # [batch_size, num_channels, seq_len]
    sensor_tensor = torch.stack(imu_values)

    return {
        "imu": sensor_tensor,
        "video": pixel_values,
    }