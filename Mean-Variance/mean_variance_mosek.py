from mosek.fusion import *
import mosek.fusion.pythonic
import numpy as np

import os
import time


"""
Purpose:
    Computes several portfolios on the optimal portfolios by

        for alpha in alphas:
            maximize   expected return - alpha * variance
            subject to the constraints

Input:
    n: Number of assets
    mu: An n dimensional vector of expected returns
    GT: A matrix with n columns so (GT')*GT  = covariance matrix
    x0: Initial holdings
    w: Initial cash holding
    alphas: List of the alphas

Output:
    The efficient frontier as list of tuples (alpha, expected return, variance)
"""
def EfficientFrontier(n,mu,GT,x0,w,lambda_risk):

    with Model("Efficient frontier") as M:
        frontier = []
        # 例如，将线程数设置为 1
        M.setSolverParam("numThreads", 10)
        # Defines the variables (holdings). Shortselling is not allowed.
        x = M.variable("x", n, Domain.greaterThan(0.0)) # Portfolio variables
        s = M.variable("s", 1, Domain.unbounded())      # Variance variable

        # Total budget constraint
        M.constraint('budget', Expr.sum(x) == w+sum(x0))

        # Computes the risk
        #M.constraint('risk', Expr.vstack(s, GT @ x) == Domain.inQCone())
        M.constraint('variance', Expr.vstack(s, 0.5, Expr.mul(GT, x)), Domain.inRotatedQCone())

        # Define objective as a weighted combination of return and variance
        alpha = M.parameter()
        M.objective('obj', ObjectiveSense.Maximize, x.T @ mu - s * alpha)
        
        # Solve multiple instances by varying the parameter alpha
        
        alpha.setValue(lambda_risk)
        M.solve()  
        start = time.time()
        M.solve()       
        print(f"time = {time.time()-start}")

        # Check if the solution is an optimal point
        solsta = M.getPrimalSolutionStatus()
        if (solsta != SolutionStatus.Optimal):
            # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
            raise Exception(f"Unexpected solution status: {solsta}")

        print(np.sort(x.level())[-1:-8:-1])
        print(np.argsort(x.level())[-1:-8:-1]+1)
        frontier.append((np.dot(mu,x.level()), s.level()[0]))

        return frontier

u_csv = "./stocks_lr_mean.csv"
cov_csv = "./cov_mat.csv"
os.environ["MOSEKLM_LICENSE_FILE"] = "./license/2400_mosek.lic"

if __name__ == '__main__':
    mu = np.loadtxt(u_csv)
    Q = np.loadtxt(cov_csv)
    n = Q.shape[0]
    w = 1.0   
    
    x0 = np.zeros(n,dtype=np.float64)
    GT = np.linalg.cholesky(Q).T
    # Some predefined alphas are chosen
    alpha = 1.0
    with mosek.Env() as env:
        assert env.getversion() == (10, 2, 11)
        frontier= EfficientFrontier(n,mu,GT,x0,w,alpha)
        print("\n-----------------------------------------------------------------------------------")
        print('Efficient frontier')
        print("-----------------------------------------------------------------------------------\n")
        print('%-12s  %-12s' % ('return','risk (variance)'))
        for i in frontier:
            print('%-12.4f  %-12.4e ' % (i[0],i[1]))




