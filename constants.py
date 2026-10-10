from dataclasses import dataclass

CAM_H, CAM_W = 375, 1242  # camera image (px), KITTI 2011_09_26 drives

@dataclass(frozen=True)
class BevParams:
    MIN_Z: float = -3.0   # BEV window: 3 m behind to 70 m ahead of the camera (m)
    MAX_Z: float = 70.0
    H: int = CAM_H        # px, same as the camera image
    W: int = CAM_W // 3   # px, a third of the camera image: 414

    @property
    def MAX_X(self):
        """Sideways, each side (m), ~40.3."""
        return self.W / 2 * (self.MAX_Z - self.MIN_Z) / self.H

BEV_PARAMS = BevParams()

EGO_WIDTH = 1.8   # VW Passat B6, the KITTI car (m)
EGO_LENGTH = 4.8
EGO_FRONT = 2.0   # front bumper ahead of the camera (m), estimate
