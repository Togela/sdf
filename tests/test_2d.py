import unittest
from sdf import *

import numpy as np
import logging

class Test2D(unittest.TestCase):
    def test_closest_surface_point(self):
        rad = 1
        geo = circle(center=[0,0], radius=rad)
        point = np.random.rand(2)

        surfpoint = geo.closest_surface_point(point)

        self.assertAlmostEqual(np.linalg.norm(surfpoint), rad, places=3)


    def test_surface_intersection(self):
        rad = 1
        geo = circle(center=[0,0], radius=rad)
        point = np.random.rand(2)
        direction = np.array([0, 1])

        surfpoint = geo.surface_intersection(point, direction)

        self.assertAlmostEqual(np.linalg.norm(surfpoint), rad, places=5, msg=f"Intersection point ({surfpoint}) not on surface (circle with r={rad})")
        vector_point_surf = (np.array(surfpoint)-np.array(point))/np.linalg.norm(np.array(surfpoint)-np.array(point))
        [self.assertAlmostEqual(c0, c1, places=5, msg=f"Intersection vector ({vector_point_surf}) different from ({direction}) doesn't work") for c0, c1 in zip(vector_point_surf, direction)]

    def test_extend_in(self):
        rad = 1
        cxy = [0, 0]
        geo = circle(center=cxy, radius=rad)
        bounds_correct = (cxy[0] - rad, cxy[0] + rad,cxy[1] - rad, cxy[1] + rad)
        
        vector = np.random.rand(2)
        vector /= np.linalg.norm(vector)
        ext = geo.extent_in(np.array(vector))
        self.assertAlmostEqual(ext, rad)

    def test_bounds(self):
        rad = 1
        cxy = [0, 0]
        geo = circle(center=cxy, radius=rad)
        bounds_correct = (cxy[0] - rad, cxy[0] + rad,cxy[1] - rad, cxy[1] + rad)
        bounds = geo.bounds

        [self.assertAlmostEqual(b, bc, places=5, msg=f"Bounds {bounds} differ from correct value {bounds_correct}") for b, bc in zip(bounds, bounds_correct)]

    def test_sample_points(self):
        rad = 1
        cxy = [0, 0]
        geo = circle(center=cxy, radius=rad)

        mindist = 0.1
        npoints = 100000
        points = geo.sample_points(npoints, mindist)
        

        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        fig, ax = plt.subplots()
        _ = ax.scatter(points[:, 0], points[:, 1])
        circles = [plt.Circle((xi, yi), radius=mindist/2, fill=False)
                for xi, yi in points]
        collection = PatchCollection(circles, match_original=True)
        ax.add_collection(collection)
        _ = ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$',
                xlim=[-rad*1.1, rad*1.1], ylim=[-rad*1.1, rad*1.1])
        ax.add_artist(plt.Circle(cxy, radius=rad, fill=False))
        plt.show(block=True)

        dist = geo(points)
        # plt.scatter(points[:,0], points[:,1]);plt.gca().set_aspect("equal");plt.show(block=True)
        [self.assertLess(d, 0) for d in dist]