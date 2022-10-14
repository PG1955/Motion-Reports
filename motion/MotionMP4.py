import os
import datetime
import cv2


def main():
    mp4 = MotionMP4('Motion')


class MotionMP4:

    def __init__(self, path, size, version, frame_rate):
        self.path = path
        self.size = size
        self.frame_rate = frame_rate
        self.version = version
        self.writer = None
        self.filename = None
        self.filepath = None

        if not os.path.exists(self.path):
            print('Creating output directory {}'.format(self.path))
            os.makedirs(self.path)
            # open(path, 'w').close()

        print('Starting process with version {}'.format(self.version))
        return

    def new_filename(self):
        self.version += 1
        self.filename = \
            str(self.version).zfill(3) + '-' + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + '.mp4'
        self.filepath = \
            os.path.join(self.path + os.sep + self.filename)
        return self.filepath

    def open(self):
        # x264 = cv2.VideoWriter_fourcc(*'X264')
        avc1 = cv2.VideoWriter_fourcc(*'AVC1')
        mp4v = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.new_filename(), mp4v, self.frame_rate, self.size)
        return self.writer

    def close(self):
        self.writer.release()
        self.writer = None

    def is_open(self):
        if self.writer:
            return True
        else:
            return False

    def get_filename(self):
        return self.filename

    def get_pathname(self):
        return self.filepath

    def get_version(self):
        return self.version


if __name__ == "__main__":
    main()
