# File: models/bayesian_discrete.py

import numpy as np

class DiscreteBayesianSkiRental:
    def __init__(self, b=10, max_day=None, prior_pmf=None):
        self.b = int(b)

        if max_day is None:
            self.max_day = 4 * self.b
        else:
            self.max_day = int(max_day)
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
        Decide the optimal day to buy (switch) skis based on the discrete prior distribution.
        Returns the day t to purchase, or max_day+1 if never purchase.
        """
        B = self.b
        M = self.max_day
        prior = self.prior_pmf.copy()

        for t in range(1, M + 1):
            Z = prior[t-1:M].sum()
            if Z <= 0:
                return M + 1  # No more purchase since all probability mass is exhausted

            post = np.zeros(M, dtype=float)
            post[t-1:M] = prior[t-1:M] / Z

            remaining_days = np.arange(t, M + 1) - t
            c = np.minimum(remaining_days, B)
            E_remain = np.dot(post[t-1:M], c)

            cost_if_buy = B
            cost_if_rent = 1 + E_remain

            if cost_if_buy <= cost_if_rent:
                return t
            # otherwise, continue renting

        return M + 1  # Does not purchase until the end

    def cost(self, decision_day, actual_days):
        """
        Compute the cost given the decision day and actual days used.
        """
        B = self.b
        if decision_day <= actual_days:
            return (decision_day - 1) + B
        else:
            return actual_days

    def competitive_ratio(self, decision_day, actual_days):
        """
        Compute the competitive ratio of the algorithm:
        algorithm cost divided by the optimal cost (min(actual_days, b)).
        """
        alg_cost = self.cost(decision_day, actual_days)
        opt_cost = min(actual_days, self.b)

        # Return meaningful ratio even if actual_days is 0
        if opt_cost <= 0:
            return float('inf') if alg_cost > 0 else 1.0

        return alg_cost / opt_cost
