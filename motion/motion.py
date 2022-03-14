"""
Motion for Python3. Emulates MotionEye but uses tne libcamera library
Arducam low light camera.

Version Date        Description
v1.10   13/01/2022  Initial version of motion.
v1.11   04/02/2022  Add post_capture parameter and buffer,
v1.13   14/02/2022  Added support for bgr colors in the ini file.
v1.14   18/02/2022  Added support for a dummy libcamaer file to allow development on windows.
v1.15   19/02/2022  Only trigger recording after the average movement level exceeds sensitivity for .
v1.16   20/02/2022  Add report motion average peak.
v1.17   23/02/2022  Add recording_trigger logic. Tidy stats and include exposure.
v1.18   23/02/2022  Add a graph panel thanks to Rune.
v1.19   25/02/2022  Coloured graph with scaling,
v1.20   26/02.2022  Caluculate scalling factor. Rename sensitivity_level trigger_point.
v1.21   27/02/2022  Rotate image 180.
v1.22   01/03/2022  Add peak movement information to statistics.
v1.23   05/03/2022  Take jpg from at the point of peak movement. 
v1.24   06/03/2022  Correct duplicate box printing.
v1.25   06/03/2022  Allow control of what is added to the jpg file. Graph and statistics.
v1.26   08/03/2022  Enlarge date and add seconds.
v1.27   09/03/2022  Allow various resolutions.
v1.28   10/03/2022  Position graph based on screen resolution.
v1.29.  11/03/2022  flip image.
v1.30   12/03/2022  Use on and off to specify boolean switches in the ini file,

"""
__author__ = "Peter Goodgame"
__name__ = "motion"
__version__ = "v1.30"

import argparse
import libcamera
from Journal import journal
import cv2
import time
import datetime
import numpy as np
import os
import sys
import signal
import configparser
import subprocess
import logging
# import random
# from systemd.journal import JournaldLogHandler
from MotionMP4 import MotionMP4


class TriggerMotion:
    sig_usr1 = False

    def __init__(self):
        if not os.name == 'nt':
            signal.signal(signal.SIGUSR1, self.trigger_motion)

    def trigger_motion(self, *args):
        self.sig_usr1 = True


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


class Graph:
    def __init__(self, g_width, g_height, boarder, g_trigger_point):
        self.b = boarder
        self.yp = 10
        self.y = int((self.yp * g_height) / 100)
        self.x = int(g_width - self.b - self.b)
        self.g_trigger_point = g_trigger_point
        self.scaling_factor = 4
        self.scaling_value = (self.y / self.scaling_factor) / trigger_point
        self.graph = np.zeros((self.y, self.x, 3), np.uint8)
        # print('Graph shape is: {}'.format(self.graph.shape))

    def update_frame(self, value):
        scaled_value = int(value * self.scaling_value)
        scaled_tp = int(self.g_trigger_point * self.scaling_value)
        log.info('Graph:scaled_value: {} trigger_point: {} '.format(scaled_value, scaled_tp))
        if scaled_value < 0:
            scaled_value = 0
        elif scaled_value >= self.y:
            scaled_value = self.y - 1
        new_graph = np.zeros((self.y, self.x, 3), np.uint8)
        new_graph[:, :-1, :] = self.graph[:, 1:, :]
        green = 0, 255, 0
        white = 255, 255, 255
        if scaled_value > scaled_tp:
            new_graph[(self.y - scaled_value):self.y - scaled_tp, -1, :] = green
            new_graph[self.y - scaled_tp:, -1, :] = white
        else:
            new_graph[self.y - scaled_value:, -1, :] = white
        self.graph = new_graph

    def get_graph(self):
        return self.graph

    def get_roi(self, g_frame):
        return g_frame[-abs(self.y + self.b):-abs(self.b), -abs(self.x + self.b):-abs(self.b),:]



def readConfiguration(signalNumber, frame):
    print('(SIGHUP) reading configuration')
    return


def display_fps(index):
    display_fps.frame_count += 1
    current = time.time()
    if current - display_fps.start >= 1:
        print("fps: {}".format(display_fps.frame_count))
        display_fps.frame_count = 0
        display_fps.start = current


def write_timestamp(wt_frame):
    # Write data and time on the video.
    wt_now = datetime.datetime.now()
    wt_text = wt_now.strftime("%Y/%m/%d %H:%M:%S")
    wt_font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 2
    line_type = cv2.LINE_AA
    text_size, _ = cv2.getTextSize(wt_text, wt_font, font_scale, thickness)
    line_height = text_size[1] + 5
    wt_pos = width - 200, height - 10
    cv2.putText(wt_frame,
                wt_text,
                wt_pos,
                wt_font,
                font_scale,
                (255, 255, 255),
                thickness, line_type)
    return wt_frame


def put_text(pt_frame, pt_text, pt_color):
    position = (10, 60)  # indent and line
    font_scale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    text_size, _ = cv2.getTextSize(pt_text, font, font_scale, thickness)
    line_height = text_size[1] + 5
    x, y0 = position
    for i, line in enumerate(pt_text.split("\n")):
        y = y0 + i * line_height
        cv2.putText(pt_frame,
                    line,
                    (x, y),
                    font,
                    font_scale,
                    pt_color,
                    thickness,
                    line_type)
    return pt_frame


def print_stats(ps_frame):
    ps_stats = 'Software version: {}\n' \
               'Exposure: {}\n' \
               'Frame rates - Record: {} Playback: {}\n' \
               'Trigger Point: {}\n' \
               'Trigger frames: {}\n' \
               'Total Frames: {}\n' \
               'Peak Movement: {}\n' \
               'Pre Movement Frames: {}\n' \
               'Post Movement Frames: {}'.format(
        __version__, exposure, record_fps, playback_fps, trigger_point, trigger_frames_to_check,
        frames_written, peak_movement, pre_frames, post_frames)
    return put_text(ps_frame, ps_stats, statistics_bgr)


def write_jpg(wj_frame):
    jpg_path = mp4.get_pathname().replace('mp4', 'jpg')
    if jpg_statistics:
        wj_frame = print_stats(wj_frame)
    if draw_jpg_graph:
        roi = graph.get_roi(wj_frame)
        roi[:] = graph.get_graph()
    print('JPEG Path: {}'.format(jpg_path))
    cv2.imwrite(jpg_path, wj_frame)


def run_cmd(rc_cmd):
    subprocess.call(rc_cmd, shell=True, executable='/bin/bash')


def get_logger():
    logger = logging.getLogger(__name__)
    # journald_handler = JournaldLogHandler()
    # journald_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    # logger.addHandler(journald_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def average_movement(am_frame, am_average):
    # Convert the image to grayscale & blur the result
    am_gray = cv2.cvtColor(am_frame, cv2.COLOR_BGR2GRAY)
    am_gray = cv2.GaussianBlur(am_gray, (21, 21), 0)

    # If the average has not yet beet calculated
    # initialise the variables and read the next frame.
    if am_average is None:
        am_average = am_gray.copy().astype("float")
        am_base_frame = am_frame
        # cam.returnFrameBuffer(data)
        return [], []

    # Accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(am_gray, am_average, 0.02)
    am_framedelta = cv2.absdiff(am_gray, cv2.convertScaleAbs(am_average))
    # Convert the difference into binary & dilate the result to fill in small holes
    am_thresh = cv2.threshold(am_framedelta, 25, 255, cv2.THRESH_BINARY)[1]
    am_thresh = cv2.dilate(am_thresh, None, iterations=2)
    # Find contours or continuous white blobs in the image
    am_contours, am_hierarchy = cv2.findContours(am_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return am_contours, am_hierarchy


def draw_box(db_frame, db_label, db_contour, db_color):
    # draw a bounding box/rectangle around the largest contour
    x, y, w, h = cv2.boundingRect(db_contour)
    cv2.rectangle(db_frame, (x, y), (x + w, y + h), db_color, 2)
    # cv2.rectangle(db_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv_area = cv2.contourArea(db_contour) Not used
    cv2.putText(db_frame, db_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, db_color, 2)
    return db_frame


def get_parameter(gp_parser, gp_name, gp_default):
    try:
        gp_ret = gp_parser.get('Motion', gp_name)
        print('{}: {}'.format(gp_name, gp_ret))
    except:
        gp_ret = gp_default
    if gp_ret == 'on':
        gp_ret = True
    elif gp_ret == 'off':
        gp_ret = False

    return gp_ret


def get_bgr(gb_str):
    red, green, blue = [int(c) for c in gb_str.split(',')]
    return (blue, green, red)


def next_index(_index, _buffer_size):
    _index += 1
    if _index >= _buffer_size:
        _index = 0
    return _index


def next_movement_index(nmi_index, nmi_buffer_size):
    nmi_index += 1
    if nmi_index >= nmi_buffer_size:
        nmi_index = 0
    return nmi_index


def Average(array):
    return round(sum(array) / len(array), 2)


if __name__ == "motion":
    software_version = __version__

    # Check for arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Debug enabled'
                        )
    args = parser.parse_args()
    # get an instance of the logger object this module will use
    log = get_logger()

    if args.debug:
        log.info('Debug os on')

    log.info('Software Version {}'.format(__version__))
    log.info('PID: {}'.format(os.getpid()))
    print('PID: {}'.format(os.getpid()))

    # Enable signals.
    killer = GracefulKiller()
    motion = TriggerMotion()

    # Read INI file.
    parser = configparser.ConfigParser()
    parser.read('motion.ini')

    record_fps = int(get_parameter(parser, 'record_fps', 24))
    playback_fps = int(get_parameter(parser, 'playback_fps', 24))
    width = int(get_parameter(parser, 'width', '640'))
    height = int(get_parameter(parser, 'height', '480'))
    trigger_point = int(get_parameter(parser, 'trigger_point', 5))
    trigger_frames_to_check = int(get_parameter(parser, 'trigger_frames_to_check', 20))
    stabilise = int(get_parameter(parser, 'stabilise', '10'))
    exposure = int(get_parameter(parser, 'exposure', '0'))
    rotate = int(get_parameter(parser, 'rotate', '0'))
    box = get_parameter(parser, 'box', 'OFF')
    draw_graph = get_parameter(parser, 'draw_graph', 'off')
    flip = bool(get_parameter(parser, 'flip', 'off'))
    draw_jpg_graph = get_parameter(parser, 'draw_jpg_graph', 'off')
    box_bgr = get_bgr(get_parameter(parser, 'box_bgr', '255,255,255'))
    box_jpg_bgr = get_bgr(get_parameter(parser, 'box_jpg_bgr', '255,0,0'))
    command = get_parameter(parser, 'command', 'None')
    pre_frames = int(get_parameter(parser, 'pre_frames', '0'))
    post_frames = int(get_parameter(parser, 'post_frames', '0'))
    grace_frames = int(get_parameter(parser, 'grace_frames', '0'))
    output_dir = get_parameter(parser, 'output_dir', '.')
    display = get_parameter(parser, 'display', 'off')
    statistics = get_parameter(parser, 'statistics', 'off')
    jpg_statistics = get_parameter(parser, 'jpg_statistics', 'off')
    statistics_bgr = get_bgr(get_parameter(parser, 'statistics_bgr', '255,255,255'))

    if args.debug:
        log.info('BOX set to: {}'.format(box))

    # Read the version ini file.
    parser = configparser.ConfigParser()
    parser.read('version.ini')
    version = int(parser.get('MP4', 'version'))

    # Enable a graph.
    graph = Graph(width, height, 10, trigger_point)

    # ===============================
    # 1) Instantiate the libcamera class
    # ===============================
    cam = libcamera.libcamera()

    """
    2) Initialize the camera
    :param1 width: Set image g_width
    :param2 height: Set image g_height
    :param3 pixelFormat: Set image format
    :param4 buffercount: Set the number of buffers
    :param5 rotation: Set the image rotation angle
    :returns ret: Whether the camera is initialized successfully
    """
    ret = cam.initCamera(width, height, libcamera.PixelFormat.RGB888, buffercount=4, rotation=rotate)
    # height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    log.info('Shape: {} X {}'.format(width, height))

    if not ret:
        # ============================
        # 3) Start Camera
        # ============================
        ret = cam.startCamera()

        frame_time = 1000000 // record_fps
        cam.set(libcamera.FrameDurationLimits, [frame_time, frame_time])

        # Initialise Variables
        size = (width, height)
        average = None
        writer = None
        stabilisation_cnt = 0
        frames_required = 0  # Frames requested for this mp4 file
        frames_written = 0  # Frames written to this mp4 file.
        peak_movement = 0  # Monitor the highest level of movement.
        log.info('Initialise MP4 output')

        # Initlalise video buffer.
        index = 0
        buffer = np.zeros((pre_frames, height, width, 3), np.dtype('uint8'))
        buffered_frame = np.zeros((1, height, width, 3), np.dtype('uint8'))
        jpg_frame = np.zeros((1, height, width, 3), np.dtype('uint8'))

        log.info('Camera started')

        mp4 = MotionMP4(output_dir, size, version, playback_fps)

        if not int(exposure) == 0:
            log.info('Set exposure to {}'.format(exposure))
            cam.set(libcamera.ExposureTime, int(exposure))

        log.info('PID: {}'.format(os.getpid()))
        # Read images and process them.
        jpg_contour = 0
        recording = False
        trigger_record = False
        trigger_point_cnt = 0
        frames_required = 0
        contour = (0, 0, 0, 0)
        stabilised = False

        # Main process loop.
        while not killer.kill_now:

            """
            Read image information

            :returns ret: Whether the image is successfully read
            :returns data: Image data information
            """
            # to run on linux - ret, data = cam.readFrame()
            # ret, data = cam.readFrame()
            # to run on Windows - ret, frame = cam.read()
            # ret, frame = cam.read()
            if not ret:
                continue

            # to run on linux use - frame = data.imageData
            # If on windows comment this out.
            # frame = data.imageData # If on windows comment this out.

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            if flip:
                frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (width, height))

            if rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)



            # Convert the image to grayscale & blur the result
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # If the average has not yet beet calculated
            # initialise the variables and read the next frame.
            if average is None:
                average = gray.copy().astype("float")
                base_frame = frame
                continue

            # Accumulate the weighted average between the current frame and
            # previous frames, then compute the difference between the current
            # frame and running average
            cv2.accumulateWeighted(gray, average, 0.02)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(average))

            # Convert the difference into binary & dilate the result to fill in small holes
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours or continuous white blobs in the image
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Stabilise the camera
            if not stabilised:
                stabilisation_cnt += 1
                if stabilisation_cnt == 1:
                    log.info('Initialisation stabilising')
                if stabilisation_cnt == stabilise - 1:
                    log.info('Shape: {}'.format(frame.shape))
                    print('Frame shape is: {}'.format(frame.shape))
                    log.info('Initialisation stabilisation completed.')
                if stabilisation_cnt < stabilise:
                    # comment out for windows, for linux - cam.returnFrameBuffer(data)
                    # cam.returnFrameBuffer(data)
                    continue
                else:
                    stabilised = True

            # Add timestamp.
            frame = write_timestamp(frame)

            if display:
                cv2.imshow('Live Data', frame)

            # ==========================================================`
            # save ts frame to the buffer and put the latest frame in bf.
            # ==========================================================`
            buffer[index] = frame

            # ------------
            # Display box.
            # ------------
            if len(contours) > 0:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                contour = contours[max_index]
                if not box == 'OFF':
                    if not contour == (0, 0, 0, 0):
                        box_text = box.replace('<value>', str(len(contours)))
                        draw_box(buffer[index], box_text, contour, box_bgr)

            index = next_index(index, pre_frames)
            buffered_frame = buffer[index]

            if args.debug:
                log.info('contour:{}'.format(contour))
                log.info('max_index:{}'.format(max_index))

            # If SIGUSR1 trigger a mp4 manually.
            if motion.sig_usr1:
                log.info('Manual SIGUSR1 detected.')
                recording = True
                frames_required = pre_frames + post_frames
                motion.sig_usr1 = False
                jpg_frame = frame

            """
            If the movement level is exceeded for n consecutive frames 
            then trigger movement.
            """
            if len(contours) > trigger_point:
                trigger_point_cnt += 1
                if trigger_point_cnt == trigger_frames_to_check:
                    trigger_record = True
                    if peak_movement < len(contours):
                        peak_movement = len(contours)
                        jpg_frame = frame
                        if not contour == (0, 0, 0, 0):
                            box_text = box.replace('<value>', str(len(contours)))
                            draw_box(jpg_frame, 'Motion', contour, box_jpg_bgr)
            else:
                trigger_point_cnt = 0

            # if trigger_record and len(contours) > trigger_point:
            if trigger_record and len(contours) > 0:
                log.info('Motion detected. contour length:{}'.format(str(len(contours))))
                if not recording:
                    recording = True
                    frames_required = post_frames + pre_frames
                else:
                    if frames_written < pre_frames:
                        frames_required = (pre_frames - frames_written) + post_frames
                    else:
                        frames_required = post_frames

            graph.update_frame(int(len(contours)))
            if draw_graph:
                roi = graph.get_roi(buffered_frame)
                roi[:] = graph.get_graph()

            if statistics and frames_required < 2:
                buffered_frame = print_stats(buffered_frame)

            if frames_required > 0:
                if display:
                    cv2.imshow('Recorded Data', buffered_frame)
                if not mp4.is_open():
                    writer = mp4.open()
                    log.info('Opening {name}...'.format(name=mp4.get_filename()))
                # jpg_frame = buffered_frame
                frames_required -= 1
                frames_written += 1
                writer.write(buffered_frame)
            else:
                if mp4.is_open():
                    journal.write('Closing {name}'.format(name=mp4.get_filename()))
                    # Write last frame. 
                    writer.write(buffered_frame)
                    mp4.close()
                    # Use the frame with the most movement jpg_frame = buffered_frame
                    write_jpg(jpg_frame)
                    frames_written = 0
                    peak_movement = 0
                    if display:
                        cv2.destroyWindow('Recorded Data')
                    if not command == "None":
                        cmd = command.replace('<MP4>', mp4.get_filename())
                        log.info('Command after replace is:{}'.format(cmd))
                        run_cmd(cmd)
                    else:
                        log.info('Command not run')

                    jpg_frame = None
                    recording = False
                    trigger_record = False
                    m_average_peak = 0
                    version += 1
                    log.info('PID: {}'.format(os.getpid()))
            """
              Return image buffer
                :param data: Send image data back
            """
            # for windows comment out linux use - cam.returnFrameBuffer(data)
            # cam.returnFrameBuffer(data)

        # Closing down.
        # cap.release()
        cam.stopCamera()
        log.info('Closing camera...')
        # cap.stopCamera()
        # cap.closeCamera()
        cv2.destroyAllWindows()

        # Update ini file.
        parser = configparser.ConfigParser()
        parser.read('version.ini')
        parser.set('MP4', 'version', str(mp4.get_version()))
        fp = open('version.ini', 'w')
        parser.write(fp)
        fp.close()

        if mp4.is_open():
            log.info('Close open MP4 file.')
            mp4.close()

    cam.closeCamera()
    log.info('Exit Motion.')
    sys.exit(0)
