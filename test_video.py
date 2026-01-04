import math

import cv2
import numpy as np
from picamera2 import Picamera2


class TagDetector:
    def __init__(
        self,
        target_ids=None,
        min_area_ratio=0.1,
        camera_index=0,
        imshow=True,
        scale=1.0,
        aligned_thres=0.5,
    ):
        self.camera_matrix = np.array(
            [[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64
        )
        self.dist_coeffs = np.zeros((5, 1))
        self.marker_length = 0.05  # unit: meter

        # support multiple target IDs
        self.target_ids = target_ids if target_ids is not None else [1, 2, 3, 4, 5]
        self.min_area_ratio = min_area_ratio

        # image display control
        self.imshow = imshow
        self.scale = scale
        self.aligned_thres = aligned_thres

        # camera initialization
        # self.cap = cv2.VideoCapture(camera_index)
        # if not self.cap.isOpened():
            # raise RuntimeError("无法打开摄像头")
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # # ArUco dictionary and parameters
        # self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # #self.params = cv2.aruco.DetectorParameters_create()
        # self.params = cv2.aruco.DetectorParameters()
        
        cam = Picamera2()
        cam.start()

    def detect(self):
        while True:
            
            print(f"====================== start to detect")
            frame = cam.capture_array()
            cv2.imshow('f', frame)

        return None

        ret, frame = self.capcap.read()
        if not ret:
            print(f"================== no image")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.params
        )

        frame_area = frame.shape[0] * frame.shape[1]
        tag_result = None

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in self.target_ids:
                    continue

                marker_corners = corners[i]
                area = cv2.contourArea(marker_corners[0])
                area_ratio = area / frame_area

                if area_ratio < self.min_area_ratio:
                    color = (0, 0, 255)  # red, not aligned
                    aligned = False
                else:
                    retval, rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(
                        marker_corners,
                        self.marker_length,
                        self.camera_matrix,
                        self.dist_coeffs,
                    )
                    rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                    sy = math.sqrt(
                        rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
                    )
                    pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    yaw = math.atan2(-rotation_matrix[2, 0], sy)

                    pitch_deg = math.degrees(pitch)
                    yaw_deg = math.degrees(yaw)
                    aligned = (
                        abs(pitch_deg) < self.aligned_thres
                        and abs(yaw_deg) < self.aligned_thres
                    )

                    color = (0, 255, 0) if aligned else (0, 0, 255)

                # draw bounding box
                cv2.polylines(
                    frame,
                    [np.int32(marker_corners)],
                    isClosed=True,
                    color=color,
                    thickness=2,
                )
                cv2.putText(
                    frame,
                    f"ID:{marker_id} Align:{aligned}",
                    tuple(marker_corners[0][0].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # return the first detected tag that meets the condition
                tag_result = Tag(tag_id=marker_id, aligned=aligned)
                break

        # display image
        if self.imshow:
            print(f"=========== show image")
            if self.scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            cv2.imshow("Aruco Tag Detection", frame)
            cv2.waitKey(1)

        return tag_result

    def start(self):
        pass

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.imshow:
            cv2.destroyAllWindows()


#if __name__ == "__main__":
    # cam = Picamera2()
    # cam.start()
    # while True:
        # print(f"====================== start to detect")
        # frame = cam.capture_array()
        # cv2.imshow('f', frame)
    # detector = TagDetector(target_ids=[1, 2, 3, 4, 5], imshow=True, scale=0.8)
    # print("start testing... press 'q' to exit")
    # while True:
        # tag = detector.detect()

        # if cv2.waitKey(1) & 0xFF == ord("q"):
            # break

    #detector.release()
    
# import cv2
# import numpy as np
# from picamera2 import Picamera2

# cam = Picamera2()
# height = 480
# width = 640
# middle = (int(width / 2), int(height / 2))
# cam.configure(cam.create_video_configuration(main={"format": 'RGB888', "size": (width, height)}))

# cam.start()
    
import cv2
import time
from picamera2 import Picamera2, MappedArray

def setupCam():
        cam = Picamera2()
        def preview(request):
            with MappedArray(request, "main") as m:
                pass

        cam.pre_callback = preview
        time.sleep(5)
        cam.start(show_preview=True)
        return cam

def test(cam):
        frame = cam.capture_array()
        height, width, _ = frame.shape
        middle = (int(width / 2), int(height / 2))
        while True:
            frame = cam.capture_array()
            cv2.circle(frame, middle, 10, (255, 0 , 255), -1)
            cv2.imshow('f', frame)

test(setupCam())
