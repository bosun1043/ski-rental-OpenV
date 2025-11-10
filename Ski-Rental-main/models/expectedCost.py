import numpy as np
from scipy.stats import norm

class ExpectedCostSkiRental:
    def __init__(self, b=10, n=5):
        self.b = b
        self.n = n  # number of predictions

    def cost(self, decision_day, actual_days):
        if decision_day <= actual_days:
            return (decision_day - 1) + self.b
        else:
            return actual_days

    def competitive_ratio(self, decision_day, actual_days):
        alg_cost = self.cost(decision_day, actual_days)
        opt_cost = min(actual_days, self.b) if actual_days > 0 else 1.0
        return alg_cost / opt_cost if opt_cost > 0 else 1.0

    def decide(self, observations, force_lambda=None, force_sigma_est=None):
        # Step 1: generate predictions
        y_bar = np.mean(observations)

        # Step 2: estimate effective std dev
        if force_sigma_est is not None:
            sigma_eff = force_sigma_est / np.sqrt(self.n)
        else:
            sigma_est_local = np.std(observations)
            sigma_eff = sigma_est_local / np.sqrt(self.n)

        # Step 3: evaluate expected cost at each candidate decision day
        max_day = 30  # you can adjust this
        days = np.arange(1, max_day + 1)

        # Standard normal CDF and survival function
        cdf = norm.cdf(days, loc=y_bar, scale=sigma_eff)
        sf = 1 - cdf

        expected_costs = (days - 1 + self.b) * sf + y_bar * cdf

        # Step 4: choose the day with the lowest expected cost
        decision_day = days[np.argmin(expected_costs)]
        return decision_day
