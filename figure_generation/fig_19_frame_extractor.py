import cv2
from pathlib import Path

def extract_nth_frame(folder, frame_n):
    """Extracts the Nth frame from each video located in a given directory, and saves them as PNG files."""

    mp4_files = sorted(Path(folder).glob("*.mp4"))

    for video_path in mp4_files:
        print(video_path.name)

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
        success, frame = cap.read()
        cap.release()

        # Build output filename: video_title_frameN.png
        output_name = f"{video_path.stem}_frame{frame_n}.png"
        output_path = video_path.parent / output_name

        # Save as PNG
        saved = cv2.imwrite(str(output_path), frame)

if __name__ == "__main__":
    path = "D:/ProtonDrive/My files/Documents/40 Projets professionnels/42 BCBL/42.05 BodyLingual/Media/Krajjat/"
    extract_nth_frame(path, 100)
