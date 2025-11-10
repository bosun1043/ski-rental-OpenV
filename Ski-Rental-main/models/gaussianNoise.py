import numpy as np

class GaussianNoiseSkiRental:
    def __init__(self, b=10, n=5, lambda_val=0.5, alpha=1.96, beta=1):
        self.b = b  # buying cost
        self.n = n  # number of predictions
        self.lambda_val = lambda_val  # lambda parameter
        self.alpha = alpha  # confidence constant
        self.beta = beta  # scaling parameter

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
        # Generate predictions with Gaussian noise
        y_bar = np.mean(observations)

        # Estimate effective standard deviation
        if force_sigma_est is not None:
            sigma_eff = force_sigma_est / np.sqrt(self.n)
        else:
            sigma_est_local = np.std(observations)
            sigma_eff = sigma_est_local / np.sqrt(self.n)

        # Adjust buying cost with confidence interval
        b_adj = self.b + self.alpha * sigma_eff

        # Use provided lambda or default
        lambda_val = force_lambda if force_lambda is not None else self.lambda_val

        if y_bar >= b_adj:
            # Case 1: Expected usage is high
            k = int(lambda_val * (self.b + self.beta * sigma_eff))
            y_sample = np.random.normal(y_bar, sigma_eff)
            
            # Calculate probabilities
            q = np.zeros(k)
            for i in range(k):
                q[i] = ((self.b - 1) / self.b) ** (k - i - 1) * (1 + sigma_eff/y_sample) / (self.b * (1 - (1 - 1/self.b)**k))
            
            # Normalize probabilities
            q = q / np.sum(q)
            
            # Sample decision day
            decision_day = np.random.choice(np.arange(1, k + 1), p=q)
        else:
            # Case 2: Expected usage is low
            l = int(np.ceil((self.b + self.beta * sigma_eff) / lambda_val))
            y_sample = np.random.normal(y_bar, sigma_eff)
            
            # Calculate probabilities
            r = np.zeros(l)
            for i in range(l):
                r[i] = ((self.b - 1) / self.b) ** (l - i - 1) * (1 - sigma_eff/y_sample) / (self.b * (1 - (1 - 1/self.b)**l))
            
            # Normalize probabilities
            r = r / np.sum(r)
            
            # Sample decision day
            decision_day = np.random.choice(np.arange(1, l + 1), p=r)

        return decision_day 