"""
label.py

Given a directory of static images to pretrain the YOLO-v5 Detector, drop into an interactive labeling loop to specify
midpoints of each object + object class, then save the corresponding paths to the `yolov5` submodule directory.

Instructions for Labeling:
    - [From this Directory] `python label.py`
    - Click on object in image (with mouse)

"""
import cv2
import os
import random


# Global Variables for Mouse (X, Y) Positions --> used for Labeling
MX, MY = 0, 0


# Mouse Callback for Labeling
def update_mouse_position(event, x, y, flags, param):
    global MX, MY
    if event == cv2.EVENT_MOUSEMOVE:
        MX, MY = x, y


def label():
    """Interactively label each workspace image with (coordinates, class) for each object."""

    # Tracking Variable
    labels = []

    # Iterate through Images
    flist, i = [x for x in os.listdir("images") if ".jpg" in x], 0
    while i < len(flist):
        # Load Image File
        print(f"Image {i + 1}/{len(flist)}")
        img = cv2.imread(os.path.join("images", flist[i]), 1)

        # Resize Image (naively scale by 1/2) so it fits on Screen
        orig = img.shape
        w, h = orig[1] // 2, orig[0] // 2
        img = cv2.resize(img, (w, h))

        # Display Image and wait for Keypress
        cv2.namedWindow("Image to Label")
        cv2.moveWindow("Image to Label", 20, 20)
        cv2.imshow("Image to Label", img)

        cv2.setMouseCallback("image", update_mouse_position())
        k = cv2.waitKey(0) & 0xFF

        # Record a Label (Class ID in 0 - 9)
        if ord("0") <= k < ord("0") + 10:
            labels.append(
                f"{int(chr(k))} {MX / w} {MY / h} 0.05625 0.1\n"
            )  # Constant Bounding-Box Size
            print(f"Recorded Class {int(chr(k))} at ({MX / w}, {MY / h})")
            print(f"Labels for Current Image:\n {''.join(labels)}")

        # (U)ndo a Label
        elif k == ord("u"):
            last = labels.pop()
            print(f"Undid --> {last}")
            print(f"Labels for Current Image:\n {''.join(labels)}")

        # Commit Labels, Move on to (N)ext Image
        elif k == ord("n"):
            # Save annotation in file
            text = "".join(labels)
            with open(os.path.join("labels", flist[i][:-3] + "txt"), "w") as f:
                f.write(text)

            # Clear
            labels, i = [], i + 1
            cv2.destroyAllWindows()

        # (Esc) to Exit
        elif k == 27:
            cv2.destroyAllWindows()
            break

        # Otherwise, just loop again
        else:
            continue


def fix_paths():
    """Store paths to each labeled image file in the top-level `yolov5` directory (to train encoder)."""

    # Get and Shuffle Files for Train/Val Split
    files = [os.path.abspath(x) for x in os.listdir("images") if ".jpg" in x]
    random.shuffle(files)

    # Fraction Training
    tfrac = (len(files) // 10) * 9

    # Create Train and Validation sets!
    train_files, val_files = files[:tfrac], files[tfrac:]

    with open(f"../yolov5/data/vla-real-train-{len(train_files)}.txt", "w") as f:
        for tf in train_files:
            f.write(tf + "\n")

    with open("../yolov5/data/vla-real-val.txt", "w") as f:
        for tf in val_files:
            f.write(tf + "\n")


if __name__ == "__main__":
    # First, interactively label each image with (coordinates, object class)
    label()

    # Then, create .txt files with pointers to images, for YOLO-v5 Training Script
    fix_paths()
