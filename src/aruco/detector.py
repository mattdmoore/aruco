from cv2 import COLOR_BGR2GRAY, aruco, cvtColor
from cv2.typing import MatLike

from aruco.marker import Marker


class Detector(aruco.ArucoDetector):
    def __init__(self) -> None:
        # ArUco Dictionary and Detector Parameters
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()

        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeMax = 50
        parameters.adaptiveThreshWinSizeStep = 5
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 10
        parameters.cornerRefinementMinAccuracy = 0.01

        super().__init__(dictionary, parameters)

    def detect_markers(self, frame: MatLike) -> dict[int, Marker]:
        frame_grayscale = cvtColor(frame, COLOR_BGR2GRAY)
        corners, ids, _ = self.detectMarkers(frame_grayscale)
        aruco.drawDetectedMarkers(frame, corners, ids)
        return {int(i): Marker(*c, *i) for c, i in zip(corners, ids)}
