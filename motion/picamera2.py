import libcamera
import cv2


class SensorFormat:
    pass


class Picamera2:
    signal_event = None

    def __init__(self):
        self.camera_config = None
        self.sensor_format = None
        self.camera = 0
        self.capture = None
        self.camera_manager = None
        self.camera_idx = None
        self.verbose_console = None
        self.log = None
        self._reset_flags = None
        self.lores_width = None
        self.lores_height = None
        self.main_width = None
        self.main_height = None

    def create_video_configuration(self, main={}, lores=None, raw=None, transform=libcamera.Transform(),
                                   colour_space=None, buffer_count=6, controls={}, display="main",
                                   encode="main") -> dict:
        """Make a configuration suitable for video recording."""
        if self.camera is None:
            raise RuntimeError("Camera not opened")
        main = self._make_initial_stream_config({"format": "XBGR8888", "size": (1280, 720)}, main)
        self.main_height = main['size'][1]
        self.main_width = main['size'][0]
        self.align_stream(main, optimal=False)
        lores = self._make_initial_stream_config({"format": "YUV420", "size": (640, 360)}, lores)
        if lores is not None:
            self.align_stream(lores, optimal=False)
            self.lores_width = lores['size'][0]
            self.lores_height = lores['size'][1]
        raw = self._make_initial_stream_config({"format": self.sensor_format, "size": main["size"]}, raw)
        if colour_space is None:
            # Choose default colour space according to the video resolution.
            if self.is_RGB(main["format"]):
                # There's a bug down in some driver where it won't accept anything other than
                # sRGB or JPEG as the colour space for an RGB stream. So until that is fixed:
                colour_space = libcamera.ColorSpace.Sycc()
            elif main["size"][0] < 1280 or main["size"][1] < 720:
                colour_space = libcamera.ColorSpace.Smpte170m()
            else:
                colour_space = libcamera.ColorSpace.Rec709()
        controls = {"NoiseReductionMode": "libcamera.controls.draft.NoiseReductionModeEnum.Fast",
                    "FrameDurationLimits": (33333, 33333)} | controls
        config = {"use_case": "video",
                  "transform": transform,
                  "colour_space": colour_space,
                  "buffer_count": buffer_count,
                  "main": main,
                  "lores": lores,
                  "raw": raw,
                  "controls": controls}
        self._add_display_and_encode(config, display, encode)
        return config


    def create_preview_configuration(self, main={}, lores=None, raw=None, transform=False,
                                     colour_space=None, buffer_count=4, controls={},
                                     display="main", encode="main"):
        config = {'use_case': 'preview',
                  'transform': "<libcamera.Transform 'identity'>",
                  'colour_space': "<libcamera.ColorSpace 'sYCC'>",
                  'buffer_count': 4,
                  'main': {'format': 'XBGR8888',
                           'size': (640, 480)},
                  'lores': None,
                  'raw': None,
                  'controls': {'NoiseReductionMode': "<NoiseReductionModeEnum.Minimal: 3>",
                               'FrameDurationLimits': (100, 83333)},
                  'display': 'main',
                  'encode': 'main'}
        return config

    def create_still_configuration(self):
        config = {'use_case': 'still',
                  'transform': "<libcamera.Transform 'identity'>",
                  'colour_space': "<libcamera.ColorSpace 'sYCC'>",
                  'buffer_count': 1,
                  'main': {'format': 'BGR888', 'size': (1920, 1080)},
                  'lores': None, 'raw': None,
                  'controls': {'NoiseReductionMode': "<NoiseReductionModeEnum.HighQuality: 2>",
                               'FrameDurationLimits': (100, 1000000000)},
                  'display': None,
                  'encode': None}
        return config

    def configure_(self, camera_config="preview") -> None:
        return None

    def start(self) -> None:
        if self.capture == None:
            self.capture = cv2.VideoCapture(0)

    def capture_array_(self, name) -> bool:
        _, image = self.capture.read()
        return image

    def capture_array(self, name="main", wait=True, signal_function=signal_event):
        _, image = self.capture.read()
        image_size = image.shape[1], image.shape[0]
        main_size = (self.main_width, self.main_height)
        lores_size = (self.lores_width, self.lores_height)
        if name == "main" and not image_size == main_size and not main_size == (None, None):
            shape = image.shape
            # print(f'Shape: {shape} Size: {main_size}')
            image = cv2.resize(image, main_size, interpolation=cv2.INTER_LINEAR)

        elif name == "lores" and not image_size == lores_size and not lores_size == (None, None):
            image = cv2.resize(image, lores_size, interpolation=cv2.INTER_LINEAR)

        if libcamera.Transform.hflip and not libcamera.Transform.vflip:
            image = cv2.flip(image, 0)
        elif libcamera.Transform.vflip and not libcamera.Transform.hflip:
            image = cv2.flip(image, 1)
        elif libcamera.Transform.vflip and libcamera.Transform.hflip:
            image = cv2.flip(image, -1)
        return image

    def configure(self, config):
        self.camera_config = config
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        return self.capture

    # def _update_stream_config(self, stream_config, libcamera_stream_config) -> None:
    #     # Update our stream config from libcamera's.
    #     stream_config["format"] = str(libcamera_stream_config.pixel_format)
    #     stream_config["size"] = (libcamera_stream_config.size.width, libcamera_stream_config.size.height)
    #     stream_config["stride"] = libcamera_stream_config.stride
    #     stream_config["framesize"] = libcamera_stream_config.frame_size

    @staticmethod
    def _make_initial_stream_config(stream_config: dict, updates: dict, ignore_list=[]) -> dict:
        """Take an initial stream_config and add any user updates.

        :param stream_config: Stream configuration
        :type stream_config: dict
        :param updates: Updates
        :type updates: dict
        :raises ValueError: Invalid key
        :return: Dictionary of stream config
        :rtype: dict
        """
        if updates is None:
            return None
        valid = ("format", "size")
        for key, value in updates.items():
            if isinstance(value, SensorFormat):
                value = str(value)
            if key in valid:
                stream_config[key] = value
            elif key in ignore_list:
                pass  # allows us to pass items from the sensor_modes as a raw stream
            else:
                raise ValueError(f"Bad key '{key}': valid stream configuration keys are {valid}")
        return stream_config

    @staticmethod
    def align_stream(stream_config: dict, optimal=True) -> None:
        if optimal:
            # Adjust the image size so that all planes are a mutliple of 32 bytes wide.
            # This matches the hardware behaviour and means we can be more efficient.
            align = 32
            if stream_config["format"] in ("YUV420", "YVU420"):
                align = 64  # because the UV planes will have half this alignment
            elif stream_config["format"] in ("XBGR8888", "XRGB8888"):
                align = 16  # 4 channels per pixel gives us an automatic extra factor of 2
        else:
            align = 2
        size = stream_config["size"]
        stream_config["size"] = (size[0] - size[0] % align, size[1] - size[1] % 2)

    @staticmethod
    def align_configuration(config: dict, optimal=True) -> None:
        Picamera2.align_stream(config["main"], optimal=optimal)
        if "lores" in config and config["lores"] is not None:
            Picamera2.align_stream(config["lores"], optimal=optimal)
        # No log_point aligning the raw stream, it wouldn't mean anything.

    @staticmethod
    def is_YUV(fmt) -> bool:
        return fmt in ("NV21", "NV12", "YUV420", "YVU420", "YVYU", "YUYV", "UYVY", "VYUY")

    @staticmethod
    def is_RGB(fmt) -> bool:
        return fmt in ("BGR888", "RGB888", "XBGR8888", "XRGB8888")

    def _add_display_and_encode(self, config, display, encode):
        pass

    def stop_recording(self):
        pass

    def close(self) -> None:
        self.capture.release()

    def check_camera_config(self, camera_config: dict) -> None:
        required_keys = ["colour_space", "transform", "main", "lores", "raw"]
        for name in required_keys:
            if name not in camera_config:
                raise RuntimeError(f"'{name}' key expected in camera configuration")

        # Check the entire camera configuration for errors.
        if not isinstance(camera_config["colour_space"], libcamera._libcamera.ColorSpace):
            raise RuntimeError("Colour space has incorrect type")
        if not isinstance(camera_config["transform"], libcamera._libcamera.Transform):
            raise RuntimeError("Transform has incorrect type")

        self.check_stream_config(camera_config["main"], "main")
        if camera_config["lores"] is not None:
            self.check_stream_config(camera_config["lores"], "lores")
            main_w, main_h = camera_config["main"]["size"]
            lores_w, lores_h = camera_config["lores"]["size"]
            if lores_w > main_w or lores_h > main_h:
                raise RuntimeError("lores stream dimensions may not exceed main stream")
            if not self.is_YUV(camera_config["lores"]["format"]):
                raise RuntimeError("lores stream must be YUV")
        if camera_config["raw"] is not None:
            self.check_stream_config(camera_config["raw"], "raw")

    def stream_configuration(self, name="main") -> dict:
        """Return the stream configuration for the named stream."""
        return self.camera_config[name]


def close(self) -> None:
    self.capture.release()
