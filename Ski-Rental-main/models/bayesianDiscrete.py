# File: models/bayesian_discrete.py
import numpy as np

class DiscreteBayesianSkiRental:
    def __init__(self, b=10, max_day=None, prior_pmf=None):
        self.b = int(b)
        self.max_day = int(4 * self.b if max_day is None else max_day)
        if self.max_day < 1:
            raise ValueError("max_day must be an integer >= 1.")

        if prior_pmf is None:
            self.prior_pmf = np.ones(self.max_day, dtype=float) / self.max_day
        else:
            arr = np.array(prior_pmf, dtype=float)
            if arr.ndim != 1:
                raise ValueError("prior_pmf must be a 1-dimensional array.")
            if len(arr) != self.max_day:
                raise ValueError(f"Length of prior_pmf ({len(arr)}) must equal max_day ({self.max_day}).")
            total = arr.sum()
            if total <= 0:
                raise ValueError("Sum of prior_pmf must be greater than 0.")
            self.prior_pmf = arr / total

    def decide(self):
        """
        Buy on the first day t such that  b <= E_rent(t),
        where E_rent(t) = sum_{k=t}^M P(T=k | T>=t) * (k - t + 1).
        Returns purchase day t, or M+1 if no purchase occurs.
        """
        B = self.b
        M = self.max_day
        prior = self.prior_pmf.copy()

        for t in range(1, M + 1):
            Z = prior[t-1:M].sum()
            if Z <= 0:
                return M + 1

            post_slice = prior[t-1:M] / Z
            remain_days = (np.arange(t, M + 1) - t + 1)
            E_rent = float(np.dot(post_slice, remain_days))

            if B <= E_rent:
                return t

        return M + 1

    def cost(self, decision_day, actual_days):
        """
        Total cost given decision day and realized horizon actual_days.
        """
        B = self.b
        if decision_day <= actual_days:
            return (decision_day - 1) + B
        else:
            return actual_days

    def competitive_ratio(self, decision_day, actual_days):
        """
        Competitive ratio = algorithm cost / optimal offline cost min(actual_days, b).
        """
        alg_cost = self.cost(decision_day, actual_days)
        opt_cost = min(actual_days, self.b)
        if opt_cost <= 0:
            return float('inf') if alg_cost > 0 else 1.0
        return alg_cost / opt_cost
