"""
This is a stub for testing Picamera2 on a windows platform.
"""


class libcamera:

    def __init__(self):
        pass


class Transform:
    hflip = None
    vflip = None

    def __init__(self, hflip=False, vflip=False):
        Transform.hflip = hflip
        Transform.vflip = vflip


print(__name__)
if __name__ == "__main__":
    transform = libcamera.Transform()
    transform = libcamera.Transform(vflip=True, hflip=True)
    print(f'hflip:{libcamera.Transform.vflip} vflip:{libcamera.Transform.hflipT}')


class ColorSpace:
    @classmethod
    def Rec709(cls):
        pass

    @classmethod
    def Sycc(cls):
        pass


def controls():
    return None
