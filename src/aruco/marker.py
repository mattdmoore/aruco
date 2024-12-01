from typing import Self

from cv2 import circle, projectPoints, solvePnP
from cv2.typing import MatLike

from aruco.constants import CAMERA_MATRIX, DIST_COEFFS


class Marker:
    def __init__(self, corners: MatLike, id: int) -> None:
        self.corners = corners
        self.id = id
        self.translation_vector = []
        self.rotation_vector = []

    def estimate_pose(self, marker: Self, frame: MatLike) -> None:
        success, rvec, tvec = solvePnP(
            objectPoints=self.corners,
            imagePoints=marker.corners,
            cameraMatrix=CAMERA_MATRIX,
            distCoeffs=DIST_COEFFS,
        )

        if success:
            self.translation_vector.append(tvec.flatten())
            self.rotation_vector.append(rvec.flatten())
            projected_real_marker, _ = projectPoints(
                self.corners,
                rvec,
                tvec,
                CAMERA_MATRIX,
                DIST_COEFFS,
            )

            for i, point in enumerate(projected_real_marker):
                circle(frame, (int(point[0][0]), int(point[0][1])), 5, (255, 0, 0), -1)
