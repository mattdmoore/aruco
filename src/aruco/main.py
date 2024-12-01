import cv2

from aruco import Detector, Marker, plot_result
from aruco.constants import EXTRINSICS

detector = Detector()
cap = cv2.VideoCapture("data/video.mp4")
extrinsics = [Marker(corners) for corners in EXTRINSICS]


def main():
    while True:
        success, frame = cap.read()
        if not success:
            break

        markers = detector.detect_markers(frame)
        for i, extrinsic in enumerate(extrinsics):
            extrinsic.estimate_pose(markers[i], frame)

        cv2.imshow("Real-World Points", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    plot_result(markers, extrinsics)


if __name__ == "__main__":
    main()
