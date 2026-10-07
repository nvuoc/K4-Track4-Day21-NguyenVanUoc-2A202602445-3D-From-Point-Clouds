"""Meaningful geometry regression checks: no model or GPU required."""
import unittest
import numpy as np
from starter.kitti_io import KittiCalib, KittiObject, load_calib
from starter.projection import velo_to_cam, cam_to_image
from src.calibration_qa import object_membership

class ProjectionChecks(unittest.TestCase):
    def test_known_synthetic_point(self):
        calib = load_calib('data/synthetic/training/calib/000000.txt')
        cam = velo_to_cam(np.array([[10., 0, 0]]), calib)
        uv, depth, mask = cam_to_image(cam,calib.P2,(375,1242,3))
        self.assertTrue(mask[0])
        np.testing.assert_allclose(depth,[9.73],atol=.05)
        np.testing.assert_allclose(uv,[[614,175]],atol=3)

    def test_invalid_depth_and_image_boundaries(self):
        P = np.array([[100.,0,50,0],[0,100,40,0],[0,0,1,0]])
        points = np.array([[0,0,2],[0,0,-1],[np.nan,0,2],[0,np.inf,2],
                           [1,0,2],[-1,0,2],[0,0,.1]])
        uv, depth, mask = cam_to_image(points,P,(80,100,3))
        np.testing.assert_array_equal(mask,[True,False,False,False,False,True,False])
        np.testing.assert_allclose(uv,[[50,40],[0,40]])
        np.testing.assert_allclose(depth,[2,2])
        empty = cam_to_image(np.empty((0,3)),P,(80,100,3))
        self.assertEqual(empty[0].shape,(0,2))

    def test_projective_denominator_is_not_camera_depth(self):
        P = np.array([[100.,0,50,0],[0,100,40,0],[0,0,1,1]])
        uv, depth, _ = cam_to_image(np.array([[0.,0,2]]),P,(80,100,3))
        np.testing.assert_allclose(uv,[[100/3,80/3]])
        np.testing.assert_allclose(depth,[2])

    def test_rigid_transform_includes_translation(self):
        calib = KittiCalib(np.eye(3,4),np.eye(3),np.column_stack((np.eye(3),[1,2,3])))
        np.testing.assert_allclose(velo_to_cam(np.array([[2,3,4]]),calib),[[3,5,7]])

    def test_oriented_bottom_center_box(self):
        obj = KittiObject('Car',0,0,0,np.array([0,0,100,80]),np.array([2,2,4]),
                          np.array([1,2,10]),np.pi/2)
        points = np.array([[1,1,11.5],[2.5,1,10],[1,2.1,10],[1,-.1,10]])
        np.testing.assert_array_equal(object_membership(points,obj),[True,False,False,False])

if __name__ == '__main__':
    unittest.main()
