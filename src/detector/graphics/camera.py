import numpy as np


class CameraIntr():
    def __init__(self, u0, v0, fx, fy, sk=0, dtype=np.float32):
        camera_xyz = np.array([
            [fx, sk, u0],
            [0, fy, v0],
            [0, 0, 1],
        ], dtype=dtype).transpose()

        pull_back_xyz = np.array([
            [1 / fx, 0, -u0 / fx],
            [0, 1 / fy, -v0 / fy],
            [0, 0, 1],
        ], dtype=dtype).transpose()

        # convert xyz -> zyx
        P = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=dtype).transpose()

        self.camera_xyz = camera_xyz
        self.pull_back_xyz = pull_back_xyz
        self.P = P
        self.camera_zyx = P @ camera_xyz @ P
        self.pull_back_zyx = P @ pull_back_xyz @ P
        self.dtype = dtype

    def xyz2uv(self, xyz, return_z=False):
        z = xyz[:, 2:]
        uv_ = xyz / z @ self.camera_xyz
        uv = uv_[:, :2]
        if return_z:
            return uv, z
        return uv

    def zyx2vu(self, zyx, return_z=False):
        z = zyx[:, :1]
        zvu = zyx / z @ self.camera_zyx
        vu = zvu[:, 1:]
        if return_z:
            return vu, z
        return vu

    def uv2xyz(self, uv, z):
        nk, *_ = uv.shape
        hom_uv = np.concatenate([uv, np.ones((nk, 1), dtype=self.dtype)], axis=1)
        xy_ = hom_uv @ self.pull_back_xyz
        xyz = z * xy_
        return xyz

    def vu2zyx(self, vu, z):
        nk, *_ = vu.shape
        hom_vu = np.concatenate([np.ones((nk, 1), dtype=self.dtype), vu], axis=1)
        _yx = hom_vu @ self.pull_back_zyx
        zyx = z * _yx
        return zyx

    def translate_camera(self, y_offset, x_offset):
        translate_xyz = np.array([
            [1, 0, x_offset],
            [0, 1, y_offset],
            [0, 0, 1],
        ], dtype=self.dtype).transpose()
        translated_xyz = self.camera_xyz @ translate_xyz
        translated_xyz = translated_xyz.transpose()
        u0 = translated_xyz[0, 2]
        v0 = translated_xyz[1, 2]
        fx = translated_xyz[0, 0]
        fy = translated_xyz[1, 1]
        sk = translated_xyz[0, 1]
        return CameraIntr(u0=u0, v0=v0, fx=fx, fy=fy, sk=sk, dtype=self.dtype)

    def scale_camera(self, y_scale, x_scale):
        scale_xyz = np.array([
            [x_scale, 0, 0],
            [0, y_scale, 0],
            [0, 0, 1],
        ], dtype=self.dtype).transpose()
        scaled_xyz = self.camera_xyz @ scale_xyz
        scaled_xyz = scaled_xyz.transpose()
        u0 = scaled_xyz[0, 2]
        v0 = scaled_xyz[1, 2]
        fx = scaled_xyz[0, 0]
        fy = scaled_xyz[1, 1]
        sk = scaled_xyz[0, 1]
        return CameraIntr(u0=u0, v0=v0, fx=fx, fy=fy, sk=sk, dtype=self.dtype)


class CameraExtr(object):
    def __init__(self, r, t, dtype=np.float32):
        _tr_concat = np.concatenate([r, t.reshape(3, 1)], axis=1)
        cam_extr_xyz = np.concatenate(
            [_tr_concat, np.zeros((1, 4))], axis=0).transpose()
        # xyzw->zyxw and vice versa
        P = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ], dtype=dtype).transpose()
        cam_extr_zyx = P @ cam_extr_xyz @ P
        self.cam_extr_xyz = cam_extr_xyz
        self.cam_extr_zyx = cam_extr_zyx

    def world_xyz2cam_xyz(self, world_xyz):
        nk, *_ = world_xyz.shape
        hom_world_xyz = np.concatenate([world_xyz, np.ones((nk, 1))], axis=1)
        hom_cam_xyz = hom_world_xyz @ self.cam_extr_xyz
        # xyzw -> xyz
        cam_xyz = hom_cam_xyz[:, :3]
        return cam_xyz

    def world_zyx2cam_zyx(self, world_zyx):
        nk, *_ = world_zyx.shape
        hom_world_zyx = np.concatenate([np.ones((nk, 1)), world_zyx], axis=1)
        hom_cam_zyx = hom_world_zyx @ self.cam_extr_zyx
        # wzyx -> zyx
        cam_zyx = hom_cam_zyx[:, 1:]
        return cam_zyx
