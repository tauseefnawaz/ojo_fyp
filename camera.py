import cv2
from imutils.video import WebcamVideoStream

class VideoCam(object):
    def __init__(self):
        self.streaming = WebcamVideoStream(src=0).start()

    def __del__(self):
        self.streaming.stop()

    def get_frame(self):
        image = self.streaming.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())

        return data