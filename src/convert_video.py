"""
Script to convert video to frames, optimize the frames with kword,
and then convert the frames back to video, joining the audio back in.
"""

import os

import cv2


def convert_video_to_png(filename, framerate):
    # Open the video file
    cap = cv2.VideoCapture(filename)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Create a directory to store the PNG files
    dirname = os.path.splitext(filename)[0]
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Loop through each frame in the video
    frame_count = 0

    # Determine original framerate
    original_framerate = cap.get(cv2.CAP_PROP_FPS)

    # depending on the target framerate, skip some frames
    frame_skip = 0
    if framerate < original_framerate:
        frame_skip = int(original_framerate / framerate) - 1

    while True:
        # Skip frames if necessary
        if frame_skip > 0:
            for i in range(frame_skip):
                cap.read()
        # Read the next frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Save the frame as a PNG file
        png_filename = os.path.join(dirname, f"frame{frame_count:04d}.png")
        cv2.imwrite(png_filename, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video file
    cap.release()

    # Extract the audio from the opened video file
    audio_filename = os.path.join(dirname, "audio.wav")
    # Determine original sample rate
    sample_rate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Extract audio
    os.system(
        f"ffmpeg -v quiet -stats -y -i '{filename}' -vn -acodec pcm_s16le -ar {sample_rate} -ac 2 '{audio_filename}'"
    )


import shutil


def cleanup_temp_dir(dirname):
    """
    Removes the temporary directory and all its contents.

    :param dirname: The name of the directory to remove.
    """
    shutil.rmtree(dirname)


def convert_png_to_video(filename, framerate):
    # The png files are in the current directory
    dirname = os.path.splitext(filename)[0]

    # Check if the directory exists
    if not os.path.exists(dirname):
        print("Error: Directory does not exist")
        return

    # Find the number of frames
    num_frames = len(os.listdir(dirname))

    # Check if there are any frames
    if num_frames == 0:
        print("Error: No frames found")
        return

    # Find the width and height of the frames (from the first file in directory)
    frame = cv2.imread(os.path.join(dirname, os.listdir(dirname)[0]))
    height, width, _ = frame.shape

    # Add "-processed" to the filename
    new_filename = os.path.splitext(filename)[0] + "-processed.mp4"

    # Create the video file
    cap = cv2.VideoWriter(
        new_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        framerate,
        (width, height),
    )

    img_list = os.listdir(dirname)
    img_list.sort()

    # Loop through each frame in the directory
    for img_filename in img_list:
        print("writing frame", img_filename, "to video")
        # Skip any files that aren't PNG files
        if not img_filename.endswith(".png"):
            continue

        # Read the frame
        frame = cv2.imread(os.path.join(dirname, img_filename))

        # Write the frame to the video file
        cap.write(frame)

    # Release the video file
    cap.release()

    # Join the audio back in
    audio_filename = os.path.join(dirname, "audio.wav")

    # We need yet another unique filename
    newer_filename = os.path.splitext(new_filename)[0] + "-2.mp4"

    os.system(
        f"ffmpeg -v quiet -stats -y -i '{new_filename}' -i '{audio_filename}' -c:v copy -c:a aac -strict experimental '{newer_filename}'"
    )


def optimize_frames(dirname, rowlength, numiterations, search, charset, init):
    """
    Optimize the frames in the given directory using kword.

    :param dirname: The name of the directory containing the frames.
    """
    os.system(
        f"python optimize.py -t '{dirname}' -o '{dirname}' -r {rowlength} -n {numiterations} -s {search} -nw True -c {charset} -i {init} -d 1"
    )


# We now have the above functions:
# convert_video_to_png
# convert_png_to_video
# optimize_frames
# cleanup_temp_dir


# We can now write the main function
def main(args):
    # Start the timer
    import time

    start_time = time.time()

    # Convert the video to PNG files
    convert_video_to_png(args.filename, args.framerate)

    # Optimize the frames
    dirname = os.path.splitext(args.filename)[0]

    # If there are more than X frames, split into multiple directories
    split_at = args.split
    if len(os.listdir(dirname)) > split_at:
        num_dirs = int(len(os.listdir(dirname)) / split_at)
        for i in range(num_dirs + 1):
            if not os.path.exists(dirname + f"-{i}"):
                os.makedirs(dirname + f"-{i}")
            for j in range(split_at):
                try:
                    shutil.move(
                        os.path.join(dirname, f"frame{(j+i*split_at):04d}.png"),
                        os.path.join(
                            dirname + f"-{i}", f"frame{(j+i*split_at):04d}.png"
                        ),
                    )
                except:
                    continue

        # Setup multiprocessing
        import multiprocessing

        # Call this function in parallel for each directory
        commands = []
        for i in range(num_dirs + 1):
            # optimize_frames(
            #     dirname + f"-{i}",
            #     args.rowlength,
            #     args.numiterations,
            #     args.search,
            #     args.charset,
            #     args.init,
            # )
            split_dirname = dirname + f"-{i}"
            commands.append(
                f"python optimize.py -t '{split_dirname}' -o '{split_dirname}' -r {args.rowlength} -n {args.numiterations} -s {args.search} -nw True -c {args.charset} -i {args.init} -d 0"
            )

        cpu_count = multiprocessing.cpu_count()
        print(f"System has {cpu_count} cores")
        print(f"Using {args.parallel} cores")

        os.system(
            # f"parallel -j {multiprocessing.cpu_count()} ::: {' '.join(commands)}"
            # Write it in the syntax separated by ; since commands have spaces
            f'(echo {"; echo ".join(commands)}) | parallel -j {args.parallel}'
            # f'({"; ".join(commands)}) | parallel'
        )

        # Join the directories back together
        for i in range(num_dirs + 1):
            for j in range(split_at):
                try:
                    shutil.move(
                        os.path.join(
                            dirname + f"-{i}", f"frame{(j+i*split_at):04d}.png"
                        ),
                        os.path.join(dirname, f"frame{(j+i*split_at):04d}.png"),
                    )
                except:
                    continue

    else:
        os.system(
            f"python optimize.py -t '{dirname}' -o '{dirname}' -r {args.rowlength} -n {args.numiterations} -s {args.search} -nw True -c {args.charset} -i {args.init} -d 0"
        )

    # Wait for 1 second for file to finish writing
    import time

    time.sleep(1)

    # Convert the PNG files back to a video
    convert_png_to_video(args.filename, args.framerate)

    # Print the time it took
    print("Time taken:", time.time() - start_time)

    # Clean up the temporary directory
    # cleanup_temp_dir(dirname)


# We can now call the main function
if __name__ == "__main__":
    # Get command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", "-f", help="The name of the video file to convert"
    )
    # frame rate
    parser.add_argument(
        "-fr",
        "--framerate",
        help="The desired frame rate of the video",
        default=10,
        type=int,
    )
    # row length
    parser.add_argument("-r", "--rowlength", help="The desired row length of the video")
    # num iterations to optimize
    parser.add_argument(
        "-n", "--numiterations", help="The number of iterations to optimize"
    )
    # search method
    parser.add_argument(
        "-s", "--search", help="The search method to use for optimization"
    )
    # charset
    parser.add_argument(
        "-c", "--charset", help="The charset to use for optimization", default="hermes"
    )
    # init mode
    parser.add_argument(
        "-i", "--init", help="The init mode to use for optimization", default="random"
    )
    # split frames at
    parser.add_argument(
        "-sp",
        "--split",
        help="The number of frames to split the video into",
        default=100,
        type=int,
    )
    # parallelism
    parser.add_argument(
        "-p",
        "--parallel",
        help="The number of processes to use for optimization",
        default=1,
    )

    args = parser.parse_args()

    # Call the main function
    main(args)

# Run the script with the following command:
# python convert_video.py -f video.mp4
