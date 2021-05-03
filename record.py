"""
record.py

Records Kinesthetic Demonstrations (physically manipulating Panda arm to demonstrate behavior) w/ optional support for
logging Web-Cam Images for Individual Demonstrations. Currently DOES NOT support Gripper manipulation!

Fully self-contained, ROS-less implementation for recording demonstrations solely using Keyboard/Joystick input.
Runs in Python3 w/ LibFranka C++ Controller.

OUTPUT FORMAT:
    - demonstrations :: List w/ 3 options:
        1) if record_vision (Command-Line Arguments) ::
            Tuple(init_frame, Array[21-dim robot joint positions (7) + velocities (7) + torques (7)])

        3) else if NOT record_vision (Command-Line Arguments) ::
            Array[21-dim robot joint positions (7) + velocities (7) + torques (7)]

Automatically saved to <args.demonstration_path>/<names.pkl>
"""
from subprocess import call, Popen
from tap import Tap

import cv2
import numpy as np
import os
import pickle
import queue
import socket
import threading
import time

# Suppress PyGame Import Text
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

# fmt: off
# CONSTANTS
STEP_TIME = 0.1                                             # Record Joint State every 0.1 seconds
LOGITECH_RESOLUTION_X, LOGITECH_RESOLUTION_Y = 1920, 1080   # Full-Resolution for Logitech Webcam


class ArgumentParser(Tap):
    # Save Parameters
    name: str                                               # Name of the Demonstrations to be Collected
    demonstration_path: str                                 # Path to Demonstration Storage Directory

    # Port & LibFranka Parameters
    robot_ip: str = "172.16.0.2"                            # IP address of the Panda Arm we're playing with!
    port: int = 8080                                        # Default Port to Open Socket between Panda & NUC

    # Webcam & Frame Logging
    record_vision: bool = True                              # Boolean whether or not to use webcam to log frames

    # Input Device
    in_device: str = "joystick"                             # Input Device -- one of < joystick | keyboard >
    # fmt: on


def connect2robot(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("localhost", PORT))
    s.listen()
    conn, addr = s.accept()
    return conn


def listen2robot(conn):
    state_length = 7 + 7 + 7
    message = str(conn.recv(1024))[2:-2]
    state_str = list(message.split(","))
    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx + 1 : idx + 1 + state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
        assert len(state_vector) == state_length
    except (AssertionError, ValueError):
        return None

    # State Vector is Concatenated Joint Positions [0 - 7], Joint Velocities [8 - 14], and Joint Torques [15 - 21]
    state_vector = np.asarray(state_vector)[0:21]
    return state_vector


def read_state(conn):
    while True:
        state_vec = listen2robot(conn)
        if state_vec is not None:
            break
    return state_vec


# Buffer-less VideoCapture -- courtesy of StackOverflow :: https://stackoverflow.com/questions/54460797/
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        assert self.cap.isOpened(), "Webcam doesn't appear to be online!"

        # Set Width and Height so we get Webcam Full Resolution
        print("[*] Setting Proper WebCam Resolutions...")
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)

        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Read frames as soon as they are available, keeping only most recent one!
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


def main():
    # Parse Arguments
    args = ArgumentParser().parse_args()

    # Setup Webcam Feed & Get Initial Frame
    if args.record_vision:
        print("[*] Setting up WebCam Feed...")
        vc = VideoCapture(0)

    # Establish Initial Connection to Robot --> First connect, then spawn LibFranka Low-Level Process (./read_State)
    print("[*] Initializing Connection to Robot and Starting Read-State Loop...")
    lf_process = Popen(f"~/libfranka/build/examples/read_State {args.robot_ip} {args.port}", shell=True)
    conn = connect2robot(args.port)

    # Drop into While Loop w/ Controlled Recording Behavior --> Keyboard Controller
    if args.in_device == "keyboard":
        print(
            "[*] Dropping into Demonstration Loop w/ Input Device Controller = Keyboard"
        )
        demonstrations = []
        while True:
            switch = input(
                "[*] Enter [q]uit or [s]tart recording or [r]eset robot ::> "
            )

            # Stop Collecting Demonstrations
            if switch == "q":
                print("[*] Exiting Interactive Loop...")
                break

            # Collect a Demonstration <----> ENSURE ROBOT IS BACK-DRIVEABLE (Hit E-Stop)
            elif switch == "s":
                # Initialize Trackers
                initial_img, demo = None, []

                # Get Initial Image (if desired)
                if args.record_vision:
                    frame = vc.read()
                    initial_img = frame

                # Drop into Collection Loop, recording States at STEP_TIME Intervals
                print(
                    "[*] Starting Demonstration %d Recording..."
                    % (len(demonstrations) + 1)
                )
                start_time = time.time()
                while True:
                    try:
                        # Read State
                        state = read_state(conn)

                        # Get Current Time and Only Record if Difference between Start and Current is > Step
                        curr_time = time.time()
                        if curr_time - start_time >= STEP_TIME:
                            demo.append(state)

                            # Reset Start Time!
                            start_time = time.time()

                    except (KeyboardInterrupt, SystemExit):
                        print("[*] Stopped Recording!")
                        break

                # Record Demo in Demonstrations
                if args.record_vision:
                    demonstrations.append([initial_img, demo])
                else:
                    demonstrations.append(demo)

                # Log
                print(f"[*] Demonstration {len(demonstrations)}\tTotal Steps: {len(demo)}")

            # Reset Robot Position <----> UNDO E-STOP FIRST
            elif switch == "r":
                # Kill Existing Socket
                conn.close()

                # Kill LibFranka Read State
                lf_process.kill()

                # Call Reset to Home Position --> libfranka/build/examples/go_JointPosition robot_ip
                call(f"~/libfranka/build/examples/go_JointPosition {args.robot_ip}", shell=True)

                # Re-initialize LF ReadState & Socket Connection
                lf_process = Popen(f"~/libfranka/build/examples/read_State {args.robot_ip} {args.port}", shell=True)
                conn = connect2robot(args.port)

            else:
                print("Invalid command: %s!" % switch)

        # Serialize and Save Demonstrations + Images
        if not os.path.exists(args.demonstration_path):
            os.makedirs(args.demonstration_path)

        with open(os.path.join(args.demonstration_path, "%s.pkl" % args.name), "wb") as f:
            pickle.dump(demonstrations, f)

        # Cleanup
        print("[*] Shutting Down...")
        lf_process.kill()
        conn.close()
        if args.record_vision:
            vc.release()
            cv2.destroyAllWindows()

    # Drop into While Loop w/ Controlled Recording Behavior --> Keyboard Controller
    elif args.in_device == "joystick":
        # Simple Joystick (Logitech Controller) Wrapper --> [A] starts recording, [B] stops recording,
        #                                                   [START] resets robots, [BACK] ends session
        class Joystick:
            def __init__(self):
                pygame.init()
                self.gamepad = pygame.joystick.Joystick(0)
                self.gamepad.init()

            def input(self):
                pygame.event.get()
                A, B = self.gamepad.get_button(0), self.gamepad.get_button(1)
                START, BACK = self.gamepad.get_button(7), self.gamepad.get_button(6)
                return A, B, START, BACK

        print("[*] Dropping into Demonstration Loop w/ Input Device Controller = Joystick\n")
        print("[*] (A) to start recording, (B) to stop, (START) to reset, and (BACK) to end...")
        joystick, demonstrations = Joystick(), []
        while True:
            start, _, reset, end = joystick.input()

            # Stop Collecting Demonstrations
            if end:
                print("[*] Exiting Interactive Loop...")
                break

            # Collect a Demonstration <----> ENSURE ROBOT IS BACK-DRIVEABLE (Hit E-Stop)
            elif start:
                # Initialize Trackers
                initial_img, demo = None, []

                # Get Initial Image (if desired)
                if args.record_vision:
                    frame = vc.read()
                    initial_img = frame

                # Drop into Collection Loop, recording States + Frames at STEP_TIME Intervals
                print(f"[*] Starting Demonstration {len(demonstrations) + 1} Recording...")
                start_time = time.time()
                while True:
                    # Read State
                    state = read_state(conn)

                    # Get Current Time and Only Record if Difference between Start and Current is > Step
                    curr_time = time.time()
                    if curr_time - start_time >= STEP_TIME:
                        demo.append(state)

                        # Reset Start Time!
                        start_time = time.time()

                    # Get Joystick Input
                    _, stop, _, _ = joystick.input()
                    if stop:
                        print("[*] Stopped Recording!")
                        break

                # Record Demo in Demonstrations
                if args.record_vision:
                    demonstrations.append([initial_img, demo])
                else:
                    demonstrations.append(demo)

                # Log
                print(f"[*] Demonstration {len(demonstrations)}\tTotal Steps: {len(demo)}")

            elif reset:
                # Kill Existing Socket
                conn.close()

                # Kill LibFranka Read State
                lf_process.kill()

                # Call Reset to Home Position --> libfranka/build/examples/go_JointPosition robot_ip
                call(f"~/libfranka/build/examples/go_JointPosition {args.robot_ip}", shell=True)

                # Re-initialize LF ReadState & Socket Connection
                lf_process = Popen(f"~/libfranka/build/examples/read_State {args.robot_ip} {args.port}", shell=True)
                conn = connect2robot(args.port)

        # Serialize and Save Demonstrations + Images
        if not os.path.exists(args.demonstration_path):
            os.makedirs(args.demonstration_path)

        with open(os.path.join(args.demonstration_path, "%s.pkl" % args.name), "wb") as f:
            pickle.dump(demonstrations, f)

        # Cleanup
        print("[*] Shutting Down...")
        lf_process.kill()
        conn.close()
        if args.record_vision:
            vc.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
