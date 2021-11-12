"""Class for the group of special linear matrices."""


import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.lie_group import MatrixLieGroup
from geomstats.geometry.lie_algebra import MatrixLieAlgebra
import numpy as np
import geomstats
import copy


class SpecialLinear(MatrixLieGroup):
    """Class for the Special Linear group SL(n).
    This is the space of invertible matrices of size n and unit determinant.
    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """
    
    def __init__(self, n):
        super(SpecialLinear, self).__init__(
            dim=int((n * (n - 1)) / 2), # George: Shouldn't dim= int(n*n -1) ? 
            n=n,
            lie_algebra=SpecialLinearLieAlgebra(n=n),
        )

        self.metric = InvariantMetric(self)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the group.
        Check the size and the value of the determinant.
        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.
        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        try: 
            point_np=np.array(point)
            n= point_np.shape
            if n == (self.n,self.n):
                det_point = np.linalg.det(point_np)
              
                if   abs(1-det_point) <= abs(atol):
                    return True 
                else:
                    return False 
            elif n==(len(point),self.n,self.n):
                   return [self.belongs(point_i) for point_i in point   ]
        except: 
          return False     


    def projection(self, point):
        """Project a point in embedding space to the group.
        This can be done by scaling the entire matrix by its determinant to
        the power 1/n.
        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in embedding manifold.
        Returns
        -------
        projected : array-like, shape=[..., n, n]
            Projected point.
        """
        ##############################################################################################################
        # George: I am not really sure that I understand which kind of projection you mean?,
        # as vector spaces or manifolds, etc?.  
        # Also, I do not see how to define this projection when det <0 and n is even. 
        # For this case, I defined my own projection. It is not unique.   
        # However, it satisfies the definition of  projection P of vector spaces: linear
        # and  P*P=P. It consists of multiplying the first column with -1 so
        # the det becomes >0.
        ###################################################################################################################
        try:
         point_np=np.array(point)
         n= point_np.shape
         if n == (self.n,self.n):
            det_point = np.linalg.det(point_np)
            
            if abs(det_point) <= abs(gs.atol):
                return None
            elif   abs(1-det_point) <= abs(gs.atol):
                return np.array(point)
            
            elif det_point > 0 :
                return (point_np)/  (np.power(det_point, 1/n[0] ))
            elif det_point < 0 and self.n % 2 != 0 :
                return (point_np)/  -(np.power(-det_point, 1/n[0] ))    
            ########################################################
            #George: Here starts the projection that I define by my own.
            ######################################################## 
            elif det_point < 0  :
                change_point  =point_np[:]
                change_point[:,0] *= -1.0
                return self.projection(change_point) 
         elif n == (len(point),self.n,self.n):
            return np.array([self.projection(point_i) for point_i in point   ])
         
        except:
          return None 


    def random_point(self, n_samples=1, bound=1.0, n_iter=100):
        """Sample in the group.
        One may use a sample from the general linear group and project it
        down to the special linear group.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0
        n_iter : int
            Maximum number of trials to sample a matrix with non zero det.
            Optional, default: 100.
        Returns
        -------
        point : array-like, shape=[..., dim]
           Sample.
        """
        n=self.n
        GenralLinearGroup= geomstats.geometry.general_linear.GeneralLinear(n=n,positive_det=True)
        random_point_general= GenralLinearGroup.random_point(n_samples, bound, n_iter)
        #George: The method GenralLinearGroup.random_point returns a matrix if  n_samples=1
        # and a list of matrices if n_samples >=2. The following if clause is to deal with 
        # the type differences.
        if n_samples ==1 :
            return self.projection(random_point_general)
        else: 
            return  [self.projection(general_point) for   general_point \
           in random_point_general]
 
        

           


class SpecialLinearLieAlgebra(MatrixLieAlgebra):
    """Class for the Lie algebra sl(n) of the Special Linear group.
    This is the space of matrices of size n with vanishing trace.
    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(SpecialLinearLieAlgebra, self).__init__(
            dim=int((n * (n - 1)) / 2), # George: Shouldn't dim be  int(n*n -1) ? 
            n=n, 
        )     
        #############################
        #George: Defining the basis:
        ############################
        basis=np.zeros(( n*n-1,n,n ))
        k=0
        for i in range(n):
            for j in range(n):
                if i !=j:
                    basis[k][i][j]=1.
                    k +=1
        for i in range(1,self.n):
            basis[n*n-n +i-1][0][0]= -1.
            basis[n*n-n +i-1][i][i]= 1.
            k =+ 1
        self.basis=basis 
    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given basis.
        Assume the basis is the one described in this answer on StackOverflow:
        https://math.stackexchange.com/a/1949395/920836
        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.
        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.
        """
        n=self.n
        Basis=self.basis
        matrix_representation_np=np.array(matrix_representation)
        basis_rep=np.zeros(matrix_representation_np.shape)
        dot_prod= lambda x:  np.sum( np.multiply(matrix_representation_np,x))
        basis_rep=[ dot_prod(Bi) for Bi  in Basis[:n*n-n]  ] + [matrix_representation_np[i][i] for i in range(1,n) ]
        #                   the elements E_{ij} for i !=j           the elements E_{ii}-E_{11} for i= 2,..,n
        return basis_rep


    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the Lie algebra.
        This method checks the shape of the input point and its trace.
        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance threshold for zero values.
        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        try: 
            point_np=np.array(point)
            n= point_np.shape
            
            if n == (self.n,self.n):
                trance_P = np.trace(point_np)
                if   abs(trance_P) <= abs(atol):
                    return True 
                else:
                    return False 
            elif n== (len(point),self.n,self.n):
                return [self.belongs(point_i) for point_i in point   ]  
        except: 
                return False  

    def random_point(self, n_samples=1, bound=1.0, n_iter=100):
        """Sample in the group.
        One may use a sample from the general linear group and project it
        down to the special linear group.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0
        n_iter : int
            Maximum number of trials to sample a matrix with non zero det.
            Optional, default: 100.
        Returns
        -------
        point : array-like, shape=[..., dim]
           Sample.
        """
        n=self.n
        GenralLinearGroup= geomstats.geometry.general_linear.GeneralLinear(n=n,positive_det=True)
        random_point_general= GenralLinearGroup.random_point(n_samples, bound, n_iter)
        if n_samples ==1 :
            return self.projection(random_point_general)
        else: 
            return  [ self.projection(ran_p_i) for ran_p_i in  random_point_general]

    def projection(self, point):
        """Project a point to the Lie algebra.
        This can be done by removing the trace in the first entry of the matrix.
        Parameters
        ----------
        point: array-like, shape=[..., n, n]
            Point.
        Returns
        -------
        point: array-like, shape=[..., n, n]
            Projected point.
        """

        point_np=np.array(point)
        n= point_np.shape
        if n == (self.n,self.n):
                proj_np=copy.deepcopy(point_np) 
                for i in range(n[0]):
                   proj_np[i][i]=0.0
                return proj_np  
        elif n == (len(point),self.n,self.n):
                return [ self.projection(point_i)  for point_i in point  ]                
            

