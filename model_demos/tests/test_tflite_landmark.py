import pytest

from cv_demos.landmark.hand_landmark_lite_1x1 import run_hand_landmark_lite_1x1
from cv_demos.landmark.palm_detection_lite_1x1 import run_palm_detection_lite_1x1
from cv_demos.landmark.pose_landmark_lite_1x1 import run_pose_landmark_lite_1x1


@pytest.mark.landmark
def test_hand_landmark_lite_1x1():
    run_hand_landmark_lite_1x1()


@pytest.mark.landmark
def test_palm_detection_lite_1x1():
    run_palm_detection_lite_1x1()


@pytest.mark.landmark
def test_pose_landmark_lite_1x1():
    run_pose_landmark_lite_1x1()
