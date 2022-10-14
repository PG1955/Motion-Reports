import cv2


def main():
    mp4 = Camera('Motion')


def PixelFormat():
    return 0


def PixelFormat():
    return 0


class Camera:

    class PixelFormat:
        RGB888 = cv2.CAP_PVAPI_PIXELFORMAT_RGB24

        def __init__(self):
            return

    class ImageData:
        def __init__(self, capture_device):
            self.capture_device = capture_device
            return

        def imageData(self):
            ret, frame = self.capture_device.read()
            return frame


    def __init__(self):
        self.cap = None
        self.ret = True
        # PixelFormat.RGB888 = None
        return

    def initCamera(self, width, height, pixel_format, buffercount=1, rotation=0):
        self.ret = False
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.buffercount = buffercount
        self.rotation = rotation
        return self.ret

    def startCamera(self):
        self.cap = cv2.VideoCapture(0)
        return True

    def readFrame(self):
        return True, Camera.ImageData(self.cap)

    def returnFrameBuffer(self, data):
        pass

    def stopCamera(self):
        pass

    def closeCamera(self):
        self.cap.release(self)


if __name__ == "__main__":
    main()
