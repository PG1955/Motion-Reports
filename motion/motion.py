"""
Motion for Python3. Emulates MotionEye but uses tne libcamera library
Arducam low light camera.

Version Date        Description
v1.10   13/01/2022  Initial version of motion.
v1.11   04/02/2022  Add post_capture parameter and buffer,
v1.13   14/02/2022  Added support for bgr colors in the ini file.
v1.14   18/02/2022  Added support for a dummy libcamera file to allow development on windows.
v1.15   19/02/2022  Only trigger recording after the average movement level exceeds sensitivity for .
v1.16   20/02/2022  Add report motion average peak.
v1.17   23/02/2022  Add recording_trigger logic. Tidy stats and include exposure.
v1.18   23/02/2022  Add a graph panel thanks to Rune.
v1.19   25/02/2022  Coloured graph with scaling,
v1.20   26/02.2022  Caluculate scalling factor. Rename sensitivity_level trigger_point.
v1.21   27/02/2022  Rotate image 180.
v1.22   01/03/2022  Add peak movement information to statistics.
v1.23   05/03/2022  Take jpg from at the log_point of peak movement.
v1.24   06/03/2022  Correct duplicate box printing.
v1.25   06/03/2022  Allow control of what is added to the jpg file. Graph and statistics.
v1.26   08/03/2022  Enlarge date and add seconds.
v1.27   09/03/2022  Allow various resolutions.
v1.28   10/03/2022  Position graph based on screen resolution.
v1.29.  11/03/2022  flip image.
v1.30   12/03/2022  Use on and off to specify boolean switches in the ini file.
v1.31   14/03/2022  Increase sensitivity and add parameter accumulateWeightedAlpha.
v1.32   18/03/2022  Performance enhancement and position date.
v1.33   09/04/2033  Add timelapse output.
v1.34   10/04/2022  Add display_frame_cnt option for test purposes.
v1.35.  01/05/2022 Correct box logic.
v1.36   19/06/2022 Only rotate the frame when writing to lessen the CPU load and reduce the core temperature.
v1.37   03/07/2022 Add output of a csv file showing peak movement values.
v1.38   12/07/2022 Log movement.
v1.39   14/07/2022 Write a csv movement record when mov
v1.40   17/07/2022 Add timings.
v2.00   20/07/2022 Eliminate the need to libcamera.
v2,01   20/07/2022 Use alternative motion detection.
v2.02   27/07/2022 Add subtraction_history parameter.
v2.03   03/08/2022 Mark the start and end of recording on the graph.
                    Also print date at the top if graph is active.
v2.04   04/08/2022 Adjust and make a pretty record and end icons on the graph.

"""
__author__ = "Peter Goodgame"
__name__ = "motion"
__version__ = "v2.04"

import argparse
# import libcamera
import math

from Journal import journal
import cv2
import time
from datetime import datetime
import numpy as np
import os
import sys
import signal
import csv
import configparser
import subprocess
import logging
# import random
# from systemd.journal import JournaldLogHandler
from MotionMP4 import MotionMP4
import multiprocessing as mp


# import openpyxl as xl
# from MotionCSV import MotionCSV


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


class MovmentTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center log_point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


class Graph:
    def __init__(self, g_width, g_height, boarder, g_trigger_point_base, g_trigger_point):
        self.b = boarder
        self.yp = 10
        self.y = int((self.yp * g_height) / 100)
        self.x = int(g_width - self.b - self.b)
        self.g_trigger_point = g_trigger_point
        self.g_trigger_point_base = g_trigger_point_base
        self.scaling_factor = 4
        self.scaling_value = (self.y / self.scaling_factor) / self.g_trigger_point
        self.graph = np.zeros((self.y, self.x, 3), np.uint8)
        self.icon_buffer_list = []
        print(f'Graph shape is: {self.graph.shape}')


    def update_frame(self, value):
        scaled_value = int(value * self.scaling_value)
        scaled_tp = int(self.g_trigger_point * self.scaling_value)
        scaled_btp = int(self.g_trigger_point_base * self.scaling_value)
        if scaled_value < 0:
            scaled_value = 0
        elif scaled_value >= self.y:
            scaled_value = self.y - 1
        new_graph = np.zeros((self.y, self.x, 3), np.uint8)
        new_graph[:, :-1, :] = self.graph[:, 1:, :]
        green = 0, 255, 0
        yellow = 0, 227, 255
        amber = 0, 183,245
        white = 255, 255, 255

        if scaled_value > scaled_tp:
            new_graph[self.y - scaled_btp:, -1, :] = white
            new_graph[self.y - scaled_tp:self.y - scaled_btp, -1, :] = amber
            new_graph[self.y - scaled_value:self.y - scaled_tp, -1, :] = green
        else:
            if scaled_value > scaled_btp:
                new_graph[self.y - scaled_btp:, -1, :] = white
                new_graph[self.y - scaled_value:self.y - scaled_btp, -1, :] = amber
            else:
                if scaled_value > 0:
                    new_graph[(self.y - scaled_value):, -1, :] = white

        if len(self.icon_buffer_list) > 0:
            top, bottom, color = self.icon_buffer_list.pop(0)
            new_graph[top: bottom, -1, :] = color
        self.graph = new_graph

    def buffer_start(self):
        for i in reversed(range(8)):
            red = 0, 0, 255
            top = int((self.y / 2)  - i)
            bottom = int((self.y / 2) + i)
            # print(f'i = {i} Top = {top} bottom = {bottom}')
            # self.icon_buffer[top :bottom, i + -1, :] = red
            self.icon_buffer_list.append([top, bottom, red])

    def buffer_end(self):
        for i in range(8):
            size = 12
            red = 0, 0, 255
            black = 0, 0, 0
            if i < 3:
                color = red
            elif i < 5:
                color = black
            elif i < 7:
                color = red
            top = int((self.y / 2)  - size / 2)
            bottom = int((self.y / 2) + size / 2)
            self.icon_buffer_list.append([top, bottom, color])

    def get_graph(self):
        return self.graph

    def get_roi(self, g_frame):
        return g_frame[-abs(self.y + self.b):-abs(self.b), -abs(self.x + self.b):-abs(self.b), :]


class MovementCSV:
    def __init__(self):
        self.trigger_point = 0
        self.frames_checked = 0
        self.subtraction_threshold = 0
        self.subtraction_history = 0
        self.trigger_value = 0  # value that triggered movement.
        self.trigger_point_base = 0  # value at which recording stops.
        self.peak_highest = 0  # Maximum movement in the reporting period.
        self.peak_average = 0  # Average movement,
        self.peak_count = 1  # number of samples.
        self.peak_total = 0  # Total movement
        self.last_write_time = datetime.now()  # Last write time.
        self.now = datetime.now()
        self.csv_file = "peakMovement.csv"
        self.columns = header = ['Timestamp', 'Trigger Point', 'Trigger Point Base', 'Frames Checked',
                                 'Subtraction Threshold', 'Subtraction History', 'Average',
                                 'Highest Peak', 'Trigger Value']
        if not os.path.isfile(self.csv_file):
            self.create()

    def create(self):
        with open(self.csv_file, 'w', newline='') as file:
            # creating a csv dict writer object
            _writer = csv.DictWriter(file, fieldnames=self.columns)
            # writing headers (field names)
            _writer.writeheader()

    def update_cvs(self, _trigger_point, _trigger_point_base, _frames_checked,
                   _subtraction_threshold, _subtraction_history,
                   _contours, _trigger_value):
        if not os.path.isfile(self.csv_file):
            self.create()

        self.trigger_point = _trigger_point
        self.trigger_point_base = _trigger_point_base
        self.frames_checked = _frames_checked
        self.subtraction_threshold = _subtraction_threshold
        self.subtraction_history = _subtraction_history
        self.trigger_value = _trigger_value

        if _contours > 0 and _trigger_value == 0:
            self.peak_count += 1
            self.peak_total += _contours

        if _contours > self.peak_highest:
            self.peak_highest = _contours

        self.now = datetime.now()
        diff = self.now - self.last_write_time
        minutes = round((diff.total_seconds() / 60), 2)

        if self.trigger_value > 0:
            self.write()
            self.trigger_value = 0

        if minutes >= 1:
            if self.peak_count > 0:
                self.peak_average = round(self.peak_total / self.peak_count)
            else:
                self.peak_average = 0
            self.write()
            self.last_write_time = self.now
            self.peak_count = 0
            self.peak_total = 0
            self.trigger_value = 0
            self.peak_highest = 0
            self.trigger_point_base

    def write(self):
        with open(self.csv_file, 'a', newline='') as file:
            _writer = csv.DictWriter(file, fieldnames=self.columns)
            timestamp = self.now.strftime("%Y-%m-%d %H:%M")
            return _writer.writerow({"Timestamp": timestamp,
                                     'Trigger Point': self.trigger_point,
                                     'Trigger Point Base': self.trigger_point_base,
                                     'Frames Checked': self.frames_checked,
                                     'Subtraction Threshold': self.subtraction_threshold,
                                     'Subtraction History': self.subtraction_history,
                                     "Average": self.peak_average,
                                     "Highest Peak": self.peak_highest,
                                     "Trigger Value": self.trigger_value})


class timingsCSV:
    """
    This class will output timing data.
    maximum is the maximum number of records written default is 100.
    """

    def __init__(self, _timings_cnt):
        self.maximum = _timings_cnt
        self.record_cnt = 0
        self.function = None
        self.seconds = None
        self.stime = None
        self.etime = None
        self.csv_file = "function-timings.csv"
        self.columns = header = ['Function', 'Seconds']
        self.delete()
        self.create()

    def delete(self):
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        self.record_cnt = 0

    def create(self):
        self.record_cnt = 0
        with open(self.csv_file, 'w', newline='') as file:
            _writer = csv.DictWriter(file, fieldnames=self.columns)
            _writer.writeheader()

    def write(self):
        if self.record_cnt < self.maximum:
            self.record_cnt += 1
            with open(self.csv_file, 'a', newline='') as file:
                _writer = csv.DictWriter(file, fieldnames=self.columns)
                _writer.writerow({"Function": self.function,
                                  'Seconds': self.seconds})
            return True
        else:
            return False

    def start(self, function):
        self.function = function
        self.stime = time.perf_counter()

    def end(self):
        self.etime = time.perf_counter()
        self.seconds = round(self.etime - self.stime, 4)
        return self.write()


class fpsCSV:
    """
    Create a csv file showing how many frames are processed each second.
    """

    def __init__(self, target_fps):
        self.maximum = 200  # Maximum number of records to write.
        self.record_cnt = 0  # Number of records written.
        self.frame_cnt = 0  # Frames processed.
        self.csv_file = "frames-per-second.csv"
        self.columns = header = ['Seconds', 'Target', 'FPS']
        self.delete()
        self.create()
        self.now = 0
        self.start = 0
        self.target_fps = target_fps

    def delete(self):
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        self.record_cnt = 0

    def create(self):
        self.record_cnt = 0
        with open(self.csv_file, 'w', newline='') as file:
            _writer = csv.DictWriter(file, fieldnames=self.columns)
            _writer.writeheader()

    def write(self):
        if self.record_cnt < self.maximum:
            self.record_cnt += 1
            with open(self.csv_file, 'a', newline='') as file:
                _writer = csv.DictWriter(file, fieldnames=self.columns)
                _writer.writerow({"Seconds": self.record_cnt,
                                  'Target': self.target_fps,
                                  'FPS': self.frame_cnt})
            self.frame_cnt = 0
            return True
        else:
            return False

    def cycle(self):
        self.frame_cnt += 1
        now = time.perf_counter()
        if self.start == 0:
            self.start = now
        duration = round(now - self.start, 2)
        if duration > 1:
            self.start = now
            return self.write()


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


def put_frame_cnt(pfc_frame, frame_count):
    # Write frame count.
    wt_font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(str(frame_count), wt_font, date_font_scale, date_font_thickness)
    boarder = 5
    line_height = text_size[1]
    line_length = text_size[0]
    wt_pos = 1 + boarder, line_height + boarder

    cv2.putText(pfc_frame,
                str(frame_count),
                wt_pos,
                wt_font,
                date_font_scale,
                date_bgr,
                date_font_thickness,
                cv2.LINE_AA)
    return pfc_frame


def put_date(wt_frame):
    # Write data and time on the video.
    wt_now = datetime.now()
    wt_text = wt_now.strftime("%Y-%m-%d %H:%M:%S")
    wt_font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(wt_text, wt_font, date_font_scale, date_font_thickness)
    boarder = 5
    line_height = text_size[1]
    line_length = text_size[0]
    if date_position == 'top' or draw_graph or draw_jpg_graph:
        wt_pos = width - line_length - boarder, line_height + boarder
    else:
        wt_pos = width - line_length - boarder, height - line_height - boarder

    cv2.putText(wt_frame,
                wt_text,
                wt_pos,
                wt_font,
                date_font_scale,
                date_bgr,
                date_font_thickness,
                cv2.LINE_AA)
    return wt_frame


def put_text(pt_frame, pt_text, pt_color):
    position = (10, 60)  # indent and line
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    text_size, _ = cv2.getTextSize(pt_text, font, statistics_font_scale, statistics_font_thickness)
    line_height = text_size[1] + 5
    x, y0 = position
    for i, line in enumerate(pt_text.split("\n")):
        y = y0 + i * line_height
        cv2.putText(pt_frame,
                    line,
                    (x, y), font,
                    statistics_font_scale,
                    pt_color,
                    statistics_font_thickness,
                    line_type)
    return pt_frame


def print_stats(ps_frame):
    ps_stats = f'Software version: {__version__}\n \
Frame rates: Record: {record_fps} Playback: {playback_fps}\n \
Trigger Point: {trigger_point} Base log_point {trigger_point_base}\n \
Trigger frames: {trigger_frames_to_check}\n \
MOG2 Subtraction Threshold: {subtraction_threshold}\n \
MOG2 Subtraction History: {subtraction_history}\n \
Total Frames: {frames_written}\n \
Peak Movement: {movement_peak} at frame number {movement_peak_frame} \n \
Pre Movement Frames: {pre_frames} Post Movement Frames: {post_frames}'
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


def write_timelapse_jpg(wtl_frame):
    timelapse_path = os.path.join(os.getcwd(), "Motion/timelapse")
    if not os.path.isdir(timelapse_path):
        os.mkdir(timelapse_path)
    timelapse_jpg = os.path.join(timelapse_path, mp4.get_filename().replace('mp4', 'jpg'))
    # jpg_path = mp4.get_pathname().replace('mp4', 'jpg')
    print('JPEG Path: {}'.format(timelapse_jpg))
    cv2.imwrite(timelapse_jpg, wtl_frame)


def run_cmd(rc_cmd):
    subprocess.call(rc_cmd, shell=True, executable='/bin/bash')


def get_logger():
    logger = logging.getLogger(__name__)
    # journald_handler = JournaldLogHandler()
    # journald_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    # logger.addHandler(journald_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def draw_box(db_frame, db_label, db_contour, db_color, db_thickness, db_fontsize):
    # draw a bounding box/rectangle around the largest contour
    x, y, w, h = cv2.boundingRect(db_contour)
    cv2.rectangle(db_frame, (x, y), (x + w, y + h), db_color, db_thickness)
    cv2.putText(db_frame, db_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, db_fontsize, db_color, db_thickness)
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
    mp.freeze_support()
    software_version = __version__

    # Check for arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Debug enabled'
                        )
    # parser.add_argument('-f', '--file',
    #                     action='store_true',
    #                     type=argparse.FileType(r),
    #                     help='mp4 file name'
    #                     )
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

    record_fps = int(get_parameter(parser, 'record_fps', 30))
    playback_fps = int(get_parameter(parser, 'playback_fps', 30))
    width = int(get_parameter(parser, 'width', '640'))
    height = int(get_parameter(parser, 'height', '480'))
    trigger_point = int(get_parameter(parser, 'trigger_point', 200))
    trigger_point_base = int(get_parameter(parser, 'trigger_point_base', 100))
    trigger_frames_to_check = int(get_parameter(parser, 'trigger_frames_to_check', 1))
    subtraction_threshold = int(get_parameter(parser, 'subtraction_threshold', 40))
    subtraction_history = int(get_parameter(parser, 'subtraction_history', 100))
    stabilise = int(get_parameter(parser, 'stabilise', '10'))
    exposure = int(get_parameter(parser, 'exposure', '0'))
    rotate = int(get_parameter(parser, 'rotate', '0'))
    box = get_parameter(parser, 'box', 'OFF')
    draw_graph = get_parameter(parser, 'draw_graph', 'off')
    box_thickness = int(get_parameter(parser, 'box_thickness', '1'))
    box_font_size = float(get_parameter(parser, 'box_font_size', '0.5'))
    flip = bool(get_parameter(parser, 'flip', 'off'))
    draw_jpg_graph = get_parameter(parser, 'draw_jpg_graph', 'off')
    box_bgr = get_bgr(get_parameter(parser, 'box_rgb', '255,255,255'))
    box_jpg_bgr = get_bgr(get_parameter(parser, 'box_jpg_rgb', '255,255,255'))
    command = get_parameter(parser, 'command', 'None')
    pre_frames = int(get_parameter(parser, 'pre_frames', '1'))
    post_frames = int(get_parameter(parser, 'post_frames', '1'))
    grace_frames = int(get_parameter(parser, 'grace_frames', '0'))
    output_dir = get_parameter(parser, 'output_dir', 'Motion')
    display = get_parameter(parser, 'display', 'off')
    display_frame_cnt = get_parameter(parser, 'display_frame_cnt', 'off')
    statistics_font_scale = float(get_parameter(parser, 'statistics_font_scale', '1.0'))
    statistics_font_thickness = int(get_parameter(parser, 'statistics_font_thickness', '1'))
    statistics = get_parameter(parser, 'statistics', 'off')
    jpg_statistics = get_parameter(parser, 'jpg_statistics', 'off')
    statistics_bgr = get_bgr(get_parameter(parser, 'statistics_rgb', '255,255,255'))
    date_position = get_parameter(parser, 'date_position', 'none')
    date_font_scale = float(get_parameter(parser, 'date_font_scale', '1.0'))
    date_font_thickness = int(get_parameter(parser, 'date_font_thickness', '1'))
    date_bgr = get_bgr(get_parameter(parser, 'date_rgb', '255,255,255'))
    jpg_timelapse_frame = int(get_parameter(parser, 'jpg_timelapse_frame', '0'))
    output_csv = get_parameter(parser, 'output_csv', 'off')
    timings_csv = get_parameter(parser, 'timings_csv', 'off')
    timings_cnt = int(get_parameter(parser, 'timings_cnt', '0'))
    roi_starting_y = int(get_parameter(parser, 'roi_starting_y', '0'))
    roi_ending_y = int(get_parameter(parser, 'roi_ending_y', '0'))
    roi_starting_x = int(get_parameter(parser, 'roi_starting_x', '-1'))
    roi_ending_x = int(get_parameter(parser, 'roi_ending_x', '-1'))

    if args.debug:
        log.info('BOX set to: {}'.format(box))

    # Instantiate movementCSV file writing.
    if output_csv:
        movement_csv = MovementCSV()

    # Read the version ini file.
    parser = configparser.ConfigParser()
    parser.read('version.ini')
    version = int(parser.get('MP4', 'version'))

    # Enable a graph.
    graph = Graph(width, height, 10, trigger_point_base, trigger_point)

    # Instatiate tracker.
    tracker = MovmentTracker()

    # Initialise capture.
    cam = cv2.VideoCapture(0)
    # picam2 = cv2.VideoCapture('Samples/bigmove.mp4')

    # Initialise Variables
    size = (width, height)
    average = None
    writer = None
    stabilisation_cnt = 0
    frames_required = 0  # Frames requested for this mp4 file
    frames_written = 0  # Frames written to this mp4 file.
    movement_peak = 0  # Monitor the highest level of movement.
    movement_peak_frame = 0  # Log the frame where peak movement occurs.
    mean_average_movement = 0  # The average amount of movement than caused a trigger record.
    object_detector = cv2.createBackgroundSubtractorMOG2(history=subtraction_history,
                                                         varThreshold=subtraction_threshold)

    log.info('Initialise MP4 output')

    # Initlalise video buffer.
    index = 0
    buffer = np.zeros((pre_frames, height, width, 3), np.dtype('uint8'))
    buffered_frame = np.zeros((1, height, width, 3), np.dtype('uint8'))
    jpg_frame = np.zeros((1, height, width, 3), np.dtype('uint8'))

    log.info('Camera started')

    mp4 = MotionMP4(output_dir, size, version, playback_fps)

    log.info('PID: {}'.format(os.getpid()))
    # Read images and process them.
    jpg_contour = 0
    recording = False
    movement_flag = False  # Set to true is movement is detected.
    movement_cnt = 0  # Number to concetutive frams that exceed the trigger log_point.
    movement_total = 0  # Total contour count for consecutive frames.
    frames_required = 0
    contour = (0, 0, 0, 0)
    resize = False
    stabilised = False

    # Instantiate timings.
    if timings_csv:
        timings = timingsCSV(timings_cnt)
        fps_cvs = fpsCSV(record_fps)

    # Main process loop.
    while not killer.kill_now:

        ret, frame = cam.read()
        if not ret:
            continue

        if display:
            key = cv2.waitKey(40)
            if key == ord('q'):
                break

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
                # comment out for windows, for q - picam2.returnFrameBuffer(data)
                # picam2.returnFrameBuffer(data)
                continue
            else:
                ah, aw, ac = frame.shape
                if aw != width or ah != height:
                    log.info('Resizing required Size: {} X {}'.format(aw, ah))
                    resize = True
                stabilised = True

        if resize:
            frame = cv2.resize(frame, (width, height))

        if flip:
            frame = cv2.flip(frame, 1)

        if rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if timings_csv:
            timings.start('Check Movement')

        # 1 Set roi.
        roi = frame[roi_starting_y: roi_ending_y, roi_starting_x: roi_ending_x]  # Bird Feeder.

        # 2 Object detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if timings_csv:
            fps_cvs.cycle()
            ret = timings.end()
            if not ret:
                break

        # Add timestamp.
        if date_position == 'top' or date_position == 'bottom':
            frame = put_date(frame)

        if jpg_timelapse_frame > 0 and \
                recording and \
                frames_written == jpg_timelapse_frame + pre_frames:
            write_timelapse_jpg(frame)

        if display:
            cv2.imshow('Live Data', frame)

        # ==========================================================`
        # save ts frame to the buffer and put the latest frame in bf.
        # ==========================================================`
        buffer[index] = frame

        # ------------
        # Display box.
        # ------------
        if len(contours) > trigger_point_base:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            contour = contours[max_index]
            if not box == 'OFF':
                if not contour == (0, 0, 0, 0):
                    box_text = box.replace('<value>', str(len(contours)))
                    draw_box(buffer[index][roi_starting_y: roi_ending_y, roi_starting_x: roi_ending_x],
                             box_text, contour, box_bgr, box_thickness, box_font_size)

        index = next_index(index, pre_frames)
        buffered_frame = buffer[index]

        if args.debug:
            log.info('contour:{}'.format(contour))
            log.info('max_index:{}'.format(max_index))

        # If SIGUSR1 trigger a mp4 manually.
        if motion.sig_usr1:
            log.info('Manual SIGUSR1 detected.')
            recording = True
            if draw_jpg_graph or draw_graph:
                graph.buffer_start()
            frames_required = pre_frames + post_frames
            motion.sig_usr1 = False
            jpg_frame = frame

        # Log movement levels for analysis.
        if output_csv:
            movement_csv.update_cvs(trigger_point,
                                    trigger_point_base,
                                    trigger_frames_to_check,
                                    subtraction_threshold,
                                    subtraction_history,
                                    len(contours),
                                    0)

        """
        If the movement level is exceeded for n consecutive frames 
        then trigger movement by setting movement to true.
        """
        c = len(contours)
        if c >= trigger_point:
            movement_cnt += 1
            movement_total += c

            if movement_peak < c:
                movement_peak = c
                movement_peak_frame = frames_written + 1
                jpg_frame = frame
                if not contour == (0, 0, 0, 0) and not box == 'OFF':
                    box_text = box.replace('<value>', str(len(contours)))
                    draw_box(buffer[index][roi_starting_y: roi_ending_y, roi_starting_x: roi_ending_x],
                             box_text, contour, box_bgr, box_thickness, box_font_size)

            if movement_cnt >= trigger_frames_to_check:
                # Log movement levels for analysis.
                if output_csv and not movement_flag:
                    movement_csv.update_cvs(trigger_point,
                                            trigger_point_base,
                                            trigger_frames_to_check,
                                            subtraction_threshold,
                                            subtraction_history,
                                            c,
                                            round(movement_total / movement_cnt))
                movement_flag = True
        else:
            if c < trigger_point_base:
                movement_flag = False
                movement_total = 0
                movement_cnt = 0

        # if trigger_record and len(contours) > trigger_point:
        # if movement_flag and len(contours) > trigger_point_base:
        if movement_flag and c > trigger_point_base:
            log.info('Motion detected. contour length:{}'.format(str(len(contours))))
            if not recording:
                recording = True
                if draw_jpg_graph or draw_graph:
                    graph.buffer_start()
                frames_required = post_frames + pre_frames
            else:
                if frames_written < pre_frames:
                    frames_required = (pre_frames - frames_written) + post_frames
                else:
                    frames_required = post_frames

        # Write Graph.
        graph.update_frame(int(len(contours)))

        if draw_graph:
            roi = graph.get_roi(buffered_frame)  # Gets the roi of the buffered frame.
            roi[:] = graph.get_graph()  # Add the graph

        if statistics and frames_required < 2:
            buffered_frame = print_stats(buffered_frame)

        if frames_required > 0:
            if display:
                cv2.imshow('Recorded Data', buffered_frame)
            if not mp4.is_open():
                writer = mp4.open()
                log.info('Opening {name}...'.format(name=mp4.get_filename()))
            if display_frame_cnt:
                buffered_frame = put_frame_cnt(buffered_frame, frames_written)
            frames_required -= 1
            frames_written += 1
            writer.write(buffered_frame)
        else:
            if mp4.is_open():
                journal.write('Closing {name}'.format(name=mp4.get_filename()))
                if draw_jpg_graph or draw_graph:
                    graph.buffer_end()
                # Write last frame.
                writer.write(buffered_frame)
                mp4.close()
                write_jpg(jpg_frame)
                frames_written = 0
                movement_peak = 0
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
                movement_flag = False
                m_average_peak = 0
                version += 1
                log.info('PID: {}'.format(os.getpid()))
        """
          Return image buffer
            :param data: Send image data back
        """
        # for windows comment out linux use - picam2.returnFrameBuffer(data)
        # picam2.returnFrameBuffer(data)

    # Closing down.
    cam.release()
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

    log.info('Exit Motion.')
    sys.exit(0)
