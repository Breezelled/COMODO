import os
import glob
import argparse
import ffmpeg

from tqdm import tqdm
from multiprocessing import Pool

def process_video(args):
    filename, fps, size, output_dir = args
    aw, ah = 0.5, 0.5
    name_clip = filename.split("/")[-1].replace(".mp4", "")

    try:
        # Initialize paths.
        processed_video_tmp_path = os.path.join(output_dir, name_clip + "-tmp.mp4")
        processed_video_path = os.path.join(output_dir, name_clip + ".mp4")
        raw_video_path = filename

        if not os.path.exists(processed_video_path):
            _, _ = (
                ffmpeg.input(raw_video_path)
                .filter("fps", fps)
                .crop(
                    "(iw - min(iw,ih))*{}".format(aw),
                    "(ih - min(iw,ih))*{}".format(ah),
                    "min(iw,ih)",
                    "min(iw,ih)",
                )
                .filter("scale", size, size)
                .output(processed_video_tmp_path)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, cmd="ffmpeg")
            )
            os.rename(processed_video_tmp_path, processed_video_path)
    except Exception as e:
        print(f"{e.stderr.decode()} processing {name_clip}")


def preprocess_videos(fps, size, video_dir, output_dir, workers):
    os.makedirs(output_dir, exist_ok=True)

    # Collect all video files.
    video_files = glob.glob(f"{video_dir}/*.mp4")
    args = [(filename, fps, size, output_dir) for filename in video_files]

    # Use multiprocessing pool to process videos in parallel.
    with Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args), total=len(args)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Downframe video to given fps and resize frames"
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        required=True,
        help="Directory with video files",
        default="path_to_videos/full_scale",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output dir for audio clips",
        default="path_to_preprocess/full_videos/processed_videos",
    )
    parser.add_argument("-f", "--fps", required=True, help="Target fps", default=10)
    parser.add_argument(
        "-s", "--size", required=True, help="Target frame size", default=224
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=48, help="Number of parallel workers."
    )

    args = parser.parse_args()

    preprocess_videos(args.fps, args.size, args.video_dir, args.output_dir, args.workers)