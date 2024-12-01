from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from cv2 import Rodrigues, undistortPoints
from cv2.typing import MatLike

from aruco.constants import CAMERA_MATRIX, DIST_COEFFS
from aruco.marker import Marker


def image_to_world(
    image_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs
):
    """
    Calculate real-world coordinates from image points.

    Parameters:
        image_points (np.ndarray): An array of 2D image points of shape (N, 2).
        rotation_vector (np.ndarray): Rotation vector (3x1 or 1x3) from the camera pose.
        translation_vector (np.ndarray): Translation vector (3x1 or 1x3) from the camera pose.
        camera_matrix (np.ndarray): Camera matrix (3x3).
        dist_coeffs (np.ndarray): Distortion coefficients (1x5 or similar).

    Returns:
        np.ndarray: An array of 3D real-world coordinates of shape (N, 3).
    """
    # Convert the image points to the appropriate shape
    image_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)

    # Undistort the image points
    undistorted_points = undistortPoints(
        image_points, camera_matrix, dist_coeffs, None, camera_matrix
    )

    # Convert rotation vector to a rotation matrix
    rotation_matrix, _ = Rodrigues(rotation_vector)

    # Calculate the inverse of the rotation matrix
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)

    # Calculate the inverse of the camera matrix
    camera_matrix_inv = np.linalg.inv(camera_matrix)

    # Prepare an empty array for the 3D points
    world_points = []

    for point in undistorted_points:
        # Back-project the 2D points to a 3D ray
        normalized_point = np.dot(
            camera_matrix_inv, np.array([point[0][0], point[0][1], 1.0])
        )

        # Calculate the scale factor to transform the ray to the world frame
        z_scale = translation_vector[2] / normalized_point[2]
        world_point_camera = normalized_point * z_scale - translation_vector.ravel()
        world_point = np.dot(rotation_matrix_inv, world_point_camera)

        world_points.append(world_point)

    return np.array(world_points)


def calculate_median_vector(extrinsics: Sequence[Marker], vector: str):
    combined_vector = np.concatenate([getattr(e, vector) for e in extrinsics])
    return np.median(combined_vector, axis=0)


def extract_points(markers: Iterable[Marker]):
    return np.concatenate([m.corners for m in markers], axis=0)


def plot_result(markers, extrinsics):
    real_marker_coords = image_to_world(
        image_points=extract_points(markers.values()),
        rotation_vector=calculate_median_vector(extrinsics, "rotation_vector"),
        translation_vector=calculate_median_vector(extrinsics, "translation_vector"),
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
    )

    plt.figure()

    def _plot_result(points: MatLike, **kwargs):
        plt.scatter(-points[:, 1], points[:, 0], **kwargs)

    _plot_result(real_marker_coords, c="red", marker="o", label="ArUco Detection")
    _plot_result(extract_points(extrinsics), c="black", marker="x", label="Extrinsics")

    plt.legend()
    plt.show()
