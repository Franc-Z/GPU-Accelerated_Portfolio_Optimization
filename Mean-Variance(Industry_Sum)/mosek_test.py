from mosek.fusion import *
import mosek.fusion.pythonic
import numpy as np

import os
import time

def EfficientFrontier(n, mu, GT, x0, w, lambda_risk, industry_labels, max_industry_weight):
    with Model("Efficient frontier") as M:
        frontier = []
        # 例如，将线程数设置为 1
        M.setSolverParam("numThreads", 1)
        # Defines the variables (holdings). Shortselling is not allowed.
        x = M.variable("x", n, Domain.greaterThan(0.0)) # Portfolio variables
        s = M.variable("s", 1, Domain.unbounded())      # Variance variable

        # Total budget constraint
        M.constraint('budget', Expr.sum(x) == w+sum(x0))

        # Computes the risk
        M.constraint('variance', Expr.vstack(s, 0.5, Expr.mul(GT, x)), Domain.inRotatedQCone())

        # Define objective as a weighted combination of return and variance
        alpha = M.parameter()
        M.objective('obj', ObjectiveSense.Maximize, x.T @ mu - s * alpha)
        
        # Add industry constraints
        # Add industry constraints
        unique_industries = np.unique(industry_labels)
        for industry in unique_industries:
            industry_indices = np.where(industry_labels == industry)[0].astype(np.int32)
            M.constraint(f'industry_{industry}', Expr.sum(x.pick(industry_indices)) <= max_industry_weight)


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

        TopK = 10
        print(np.sort(x.level())[-1:-TopK:-1])
        print(np.argsort(x.level())[-1:-TopK:-1]+1)
        frontier.append((np.dot(mu,x.level()), s.level()[0]))

        return frontier

u_csv = "/nvtest/stocks_lr_mean.csv"
cov_csv = "/nvtest/cov_mat.csv"
os.environ["MOSEKLM_LICENSE_FILE"] = "/nvtest/mosek_lic/mosek.lic"

if __name__ == '__main__':
    mu = np.loadtxt(u_csv)
    Q = np.loadtxt(cov_csv)
    n = Q.shape[0]
    w = 1.0   
    
    x0 = np.zeros(n, dtype=np.float64)
    GT = np.linalg.cholesky(Q).T

    # Example industry labels (replace with actual industry labels)
    industry_labels = np.random.randint(1, 41, size=n,dtype=np.int32)
    
    # Maximum weight for each industry (e.g., 10%)
    max_industry_weight = 0.10

    # Some predefined alphas are chosen
    alpha = 1.0
    with mosek.Env() as env:
        assert env.getversion() == (10, 2, 13)
        frontier = EfficientFrontier(n, mu, GT, x0, w, alpha, industry_labels, max_industry_weight)
        print("\n-----------------------------------------------------------------------------------")
        print('Efficient frontier')
        print("-----------------------------------------------------------------------------------\n")
        print('%-12s  %-12s' % ('return','risk (variance)'))
        for i in frontier:
            print('%-12.4f  %-12.4e ' % (i[0],i[1]))
