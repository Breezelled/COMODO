import os
import json
import argparse
import ffmpeg
from tqdm import tqdm
from multiprocessing import Pool

dataset_dir = "your_dataset_path"


def process_video(args):
    folder, target_fps, target_size, json_map = args
    # Construct the full root directory
    full_root = os.path.join(dataset_dir, "takes", folder)
    if full_root not in json_map:
        print(f"Warning: {full_root} is not in the JSON data!")
        return

    entry = json_map[full_root]
    frame_aligned = entry.get("frame_aligned_videos", {})
    aria_key = None
    for key in frame_aligned.keys():
        if key.lower().startswith("aria"):
            aria_key = key
            break
    if aria_key is None:
        print(
            f"No key starting with 'aria' found in frame_aligned_videos for {full_root}"
        )
        return

    try:
        rel_path_orig = frame_aligned[aria_key]["rgb"]["relative_path"]
    except KeyError as ke:
        print(f"Missing key in {full_root}: {ke}")
        return

    parts = rel_path_orig.split("/")
    if len(parts) < 2:
        print(f"Unexpected relative_path format for {full_root}: {rel_path_orig}")
        return

    # Construct the new relative_path:
    # Append "downscaled/448/" after the first part, then add the second part
    new_rel_path = os.path.join(parts[0], "downscaled", "448", parts[1])
    # Full path to the video file
    video_path = os.path.join(full_root, new_rel_path)

    # Use the folder name as the clip name
    name_clip = folder
    processed_video_tmp_path = os.path.join(full_root, name_clip + "-tmp.mp4")
    processed_video_path = os.path.join(full_root, name_clip + ".mp4")

    try:
        if not os.path.exists(processed_video_path):
            # Use ffmpeg to preprocess the video:
            # Change fps, crop centrally, and scale to the target size.
            aw, ah = 0.5, 0.5
            _, _ = (
                ffmpeg.input(video_path)
                .filter("fps", target_fps)
                .crop(
                    "(iw - min(iw,ih))*{}".format(aw),
                    "(ih - min(iw,ih))*{}".format(ah),
                    "min(iw,ih)",
                    "min(iw,ih)",
                )
                .filter("scale", target_size, target_size)
                .output(processed_video_tmp_path)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, cmd="ffmpeg")
            )
            os.rename(processed_video_tmp_path, processed_video_path)
    except Exception as e:
        # If the exception contains stderr information, print it out.
        print(f"Error processing {name_clip}: {e}")


def main(target_fps, target_size, workers):
    # Load the takes.json data
    json_file = dataset_dir + "takes.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build a mapping from full root_dir to the complete entry.
    # Adjust the keys by prepending the global dataset_dir.
    json_map = {os.path.join(dataset_dir, entry["root_dir"]): entry for entry in data}

    # Read folder names from folders_with_categories.txt (first column, format: "folder\tcategory")
    input_file = dataset_dir + "folders_with_categories.txt"
    folders = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                folders.append(parts[0])

    # Construct a list of tasks for processing
    args_list = [(folder, target_fps, target_size, json_map) for folder in folders]

    # Use a multiprocessing pool to process videos in parallel
    with Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args_list), total=len(args_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess egoexo4d videos")
    parser.add_argument(
        "-f", "--fps", type=int, default=10, help="Target fps (e.g., 10)"
    )
    parser.add_argument(
        "-s", "--size", type=int, default=224, help="Target frame size (e.g., 224)"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes",
    )
    args = parser.parse_args()

    main(args.fps, args.size, args.workers)
