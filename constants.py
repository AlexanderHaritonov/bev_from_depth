from dataclasses import dataclass

@dataclass(frozen=True)
class BevParams:
    MIN_Z: float = -3.0   # BEV window: 3 m behind to 70 m ahead of the camera (m)
    MAX_Y: float = 70.0
    MAX_X: float = 50.0   # sideways, each side (m)
    H: int = 375          # px, same as the camera image

BEV_PARAMS = BevParams()

EGO_WIDTH = 1.8   # VW Passat B6, the KITTI car (m)
EGO_LENGTH = 4.8
EGO_FRONT = 2.0   # front bumper ahead of the camera (m), estimate
