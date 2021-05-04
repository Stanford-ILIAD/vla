"""
teleoperate.py

Code for using pre-trained YoloV5 Detector + Latent Action Autoencoder for assistive teleoperation of a Franka Emika
Panda arm.
"""
from argparse import Namespace
from tap import Tap

from src.models import YOLOCAE
from src.preprocessing import letterbox
from yolo import YOLODetector

import socket

import cv2
import numpy as np
import os
import time
import torch

# Suppress PyGame Import Text
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


# Ground-Truth Checkpoints for Latent Action Model -- TODO :: Add model path here!
CHECKPOINT = None

# End-Effector Map
END_EFF_DIM = 6


class ArgumentParser(Tap):
    # fmt: off

    # YOLO-v5 Parameters
    model: str = "la"                                           # Teleoperation mode in <endeff | la (latent actions)>
    yolo_model: str = None                                      # Path to in-domain pretrained YOLO-v5 Detector
    n_classes: int = -1                                         # Number of Total Object Classes

    # Control Parameters
    latent_dim: int = 1                                         # Dimensionality of Latent Space (match Joystick input)
    state_dim: int = 7                                          # Dimensionality of Robot State (7-DoF Joint Angles)

    # Tunable Parameters
    action_scale: float = 1.0                                   # Scalar for increasing ||joint velocities|| (actions)

    # CAE Model Parameters
    hidden: int = 30                                            # Size of AutoEncoder Hidden Layer (Dylan Magic)

    # End2End CAE Model Parameters
    feat_dim: int = 32                                          # Size to Compress CNN Representation (Embedding)

    # Preprocessing Parameters
    img_size: int = 640                                         # Resized Image Size (default: 640 x 640)

    # Training Parameters -- necessary for model restoration
    epochs: int = 1000                                          # Number of training epochs to run
    bsz: int = 6000                                             # Batch Size for training
    lr: float = 0.01                                            # Learning Rate for training
    lr_step_size: int = 400                                     # How many epochs to run before LR decay
    lr_gamma: float = 0.1                                       # Learning Rate Gamma (decay rate)


class JoystickControl(object):
    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1

    def input(self):
        pygame.event.get()
        z = self.gamepad.get_axis(0)
        if abs(z) < self.DEADBAND:
            z = 0.0
        a, stop = self.gamepad.get_button(0), self.gamepad.get_button(7)

        return [z], a, stop


class Model(object):
    def __init__(self, hparams, detector):
        self.hparams, self.detector = hparams, detector
        self.model = YOLOCAE(hparams, detector)

        # Get Checkpoint File
        model_dict = torch.load(CHECKPOINT, map_location='cpu')
        self.model.load_state_dict(model_dict['state_dict'])
        self.model.eval()

        self.detected, self.class_encoding = False, None

    def decoder(self, img, s, z):
        img = letterbox(img, 640)                           # Resize
        img = img[:, :, ::-1].copy()                        # BGR to RGB
        img = np.transpose(img, (2, 0, 1))                  # Channel axis first
        img = torch.FloatTensor([img]) / 256.0              # Normalize to [0, 1]

        # Create CAE Embedding
        s, z = torch.FloatTensor([s]), torch.FloatTensor([z[:self.model.latent_dim]])
        context = (img, s, z)

        # Save detection computation by not processing every frame
        if not self.detected:
            self.model.store_visual(img)
            self.detected = True

        if self.model.stored is None:
            print('[*] No initial Object Detection. Standing by...')
            while True:
                pass

        a_tensor = self.model.decoder(context, use_stored=True)
        a_numpy = a_tensor.detach().numpy()[0]
        return list(a_numpy)


# End-Effector Control Functions (Resolved Rates)
def resolved_rates(xdot, j, scale=1.0):
    """ Compute the pseudo-inverse of the Jacobian to map the delta in end-effector velocity to joint velocities """
    j_inv = np.linalg.pinv(j)
    return [qd * scale for qd in (np.dot(j_inv, xdot) + np.random.uniform(-0.01, 0.01, 7))]


# Robot Control Functions
def connect2robot(PORT):
    """ Open a Socket Connection to the Low-Level Controller """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('localhost', PORT))
    s.listen()
    conn, addr = s.accept()
    return conn


def send2robot(conn, qdot, limit=1.5):
    """ Send a Joint Velocity Command to the Low Level Controller """
    qdot = np.asarray(qdot)
    scale = np.linalg.norm(qdot)
    if scale > limit:
        qdot = np.asarray([qdot[i] * limit / scale for i in range(7)])
    send_msg = np.array2string(qdot, precision=5, separator=',', suppress_small=True)[1:-1]
    send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())


def listen2robot(conn):
    """ Read in the State Information Sent Back by Low Level Controller """
    state_length = 7 + 7 + 7 + 42
    state_message = str(conn.recv(2048))[2:-2]
    state_str = list(state_message.split(","))
    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx + 1:idx + 1 + state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
        assert(len(state_vector) == state_length)
    except (ValueError, AssertionError):
        return None

    state = {
        'q': np.asarray(state_vector[0:7]),
        'dq': np.asarray(state_vector[7:14]),
        'tau': np.asarray(state_vector[14:21]),
        'J': np.array(state_vector[21:]).reshape(7, 6).T
    }

    return state


def readState(conn):
    """ Read Full State Info """
    while True:
        state = listen2robot(conn)
        if state is not None:
            break
    return state


# Main Teleoperation Code
def main():
    # Parse Arguments
    print('[*] Starting up...')
    args = Namespace(**ArgumentParser().parse_args().as_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\t[*] \"I'm rooting for the machines.\" (Claude Shannon)")

    # Connect to Gamepad, Robot
    print('\n[*] Connecting to Gamepad...')
    joystick = JoystickControl()

    print('[*] Connecting to Low-Level Controller...')
    conn = connect2robot(8080)

    print('[*] Setting up WebCam Feed...')
    vc = cv2.VideoCapture(0)
    assert vc.isOpened(), "Webcam doesn't appear to be online!"

    print('[*] Setting Proper WebCam Resolutions...')
    vc.set(3, 1920)
    vc.set(4, 1080)

    # Load Latent Actions Model -- otherwise, we're in "endeff" (End-Effector Control) mode!
    if args.model == "la":
        print('\n[*] Loading Model...')
        detector = YOLODetector(args.yolo_model, device)
        model = Model(args, detector)

    # Enter Control Loop
    detected, trajectory, inputs, start_time, toggle, pressed, pressed_time = False, [], [], time.time(), 0, False, None
    try:
        while True:
            # Read the Robot State
            state = readState(conn)

            # Append State to Trajectory
            trajectory.append(state['q'])

            # Measure Joystick Input
            z, a, stop = joystick.input()
            if stop:
                print("[*] You pressed START, so saving & exiting...")
                break

            # Add z to inputs
            inputs.append(z)

            # Latent Actions Model
            if args.model == "la":
                # Read BGR image from webcam
                if not detected:
                    all_ok, frame = vc.read()
                    assert all_ok, "Webcam stopped working?"
                    detected = True

                # Decode latent action
                qdot = [vel * args.action_scale for vel in model.decoder(frame, state['q'], z)]

                # Send joint-velocity command
                send2robot(conn, qdot)

            # Otherwise - End Effector Control
            else:
                # If A-Button Pressed, switch mode
                if a:
                    if not pressed:
                        pressed, pressed_time = True, time.time()
                        toggle = (toggle + 1) % END_EFF_DIM

                    # Make sure not "holding" button on press
                    if time.time() - pressed_time > 0.25:
                        pressed = False

                # Otherwise --> Control w/ Latent Dim
                else:
                    xdot = np.zeros(END_EFF_DIM)
                    xdot[toggle] = z[0]

                    # Resolved Rate Motion Control
                    qdot = resolved_rates(xdot, state['J'], scale=args.action_scale / 2)

                    # Send joint-velocity command
                    send2robot(conn, qdot)

    except (KeyboardInterrupt, ConnectionResetError, BrokenPipeError):
        # Just don't crash the program on Ctrl-C or Socket Error (Controller Death)
        pass


if __name__ == "__main__":
    main()
