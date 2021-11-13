"""Unit tests for the Special Linear group."""

import warnings
import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from special_linear import SpecialLinear, SpecialLinearLieAlgebra


class TestSpecialLinear(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        self.n = 3
        self.n_samples = 2
        self.group = SpecialLinear(n=self.n)
        self.algebra = SpecialLinearLieAlgebra(n=self.n)

        warnings.simplefilter("ignore", category=ImportWarning)

    def test_belongs(self):
        point = gs.array([[[1.0, 0.0, 0.0],[0.0,2.0,0.0],[0.0,0.0,0.5]]])
        result = self.group.belongs(point)
        expected = True
        self.assertAllClose(result, expected)



    def test_random_and_belongs(self):
        random_points=self.group.random_point(n_samples=self.n_samples)
        if self.n_samples ==1 :
            memberships=[self.group.belongs(random_points)]
        else:
            memberships= [self.group.belongs(random_point_i) for  random_point_i in random_points]
            expected = [True]*self.n_samples
            self.assertAllClose(memberships, expected)

    def test_projection_and_belongs(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.group, shape)
        for res in result:
            self.assertTrue(res)

    def test_belongs_algebra(self):
        point = gs.array([[4.0, 3.1, 2.0],[2.0,2.0,-1.0],[1.0,-3.0,-3.0]])
        result = self.algebra.belongs(point)
        expected = False
        self.assertAllClose(result, expected)

    def test_random_and_belongs_algebra(self):
        GenralLinearGroup= geomstats.geometry.general_linear.GeneralLinear(n=self.n,positive_det=True)
        random_points= GenralLinearGroup.random_point(n_samples=self.n_samples)
        if self.n_samples ==1 :
            projected = self.algebra.projection(random_points)
            memberships=[self.algebra.belongs(projected)]
        else:
            projected = [self.algebra.projection(random_point_i) for \
                 random_point_i in random_points]
            memberships= [self.algebra.belongs(random_point_i) for \
                 random_point_i in projected]
            expected = [True]*self.n_samples
            self.assertAllClose(memberships, expected)

    def test_projection_and_belongs_algebra(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.algebra, shape)
        for res in result:
            self.assertTrue(res)




