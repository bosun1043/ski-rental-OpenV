import numpy as np

class MultiGaussianAdaptedSkiRental:
    """
    A multi-prediction version of the Gaussian-adapted Ski Rental algorithm.
    We generate n noisy predictions, average them, and use sigma_eff = sigma / sqrt(n).
    """
    def __init__(self, b=10, n=5):
        """
        Initialize the algorithm.

        Parameters:
        b (int): cost to purchase skis
        n (int): number of noisy predictions to average
        """
        self.b = b
        self.n = n  # number of Gaussian predictions per trial

    def set_parameters(self, sigma_eff):
        """
        Configure algorithm parameters based on the effective standard deviation.

        Parameters:
        sigma_eff (float): effective standard deviation after averaging multiple predictions

        Returns:
        tuple: (lambda_val, alpha, beta, b_adj)
            - lambda_val: mixing parameter
            - alpha: confidence bound multiplier
            - beta: noise adjustment factor
            - b_adj: adjusted buy-threshold (b + alpha * sigma_eff)
        """
        # λ parameter (clamped to [0,1])
        lambda_val = min(1, np.sqrt(self.b / (self.b + sigma_eff))) if sigma_eff > 0 else 1

        # α parameter (clamped to [0,1.96])
        alpha = min(1.96, np.sqrt(2 * np.log(self.b / sigma_eff))) if sigma_eff > 0 else 1.96

        # β parameter (clamped to [0,1])
        beta = min(1, 2 * sigma_eff / self.b) if sigma_eff > 0 else 0

        # Adjusted threshold for buying
        b_adj = self.b + alpha * sigma_eff
        return lambda_val, alpha, beta, b_adj

    def cost(self, decision_day, actual_days):
        """
        Compute the incurred cost given the decision day and actual usage.

        Parameters:
        decision_day (int): the day on which we decide to purchase
        actual_days (int): the actual number of days skis are needed

        Returns:
        float: total cost
        """
        if decision_day <= actual_days:
            # We rented until the purchase day, then bought
            return (decision_day - 1) + self.b
        else:
            # We never bought; just rented for actual_days
            return actual_days

    def competitive_ratio(self, decision_day, actual_days):
        """
        Compute the competitive ratio of the algorithm:
        (algorithm cost) / (optimal offline cost).

        Parameters:
        decision_day (int): the day on which we decide to purchase
        actual_days (int): the actual number of days skis are needed

        Returns:
        float: competitive ratio
        """
        alg_cost = self.cost(decision_day, actual_days)
        opt_cost = min(actual_days, self.b) if actual_days > 0 else 1.0
        return alg_cost / opt_cost if opt_cost > 0 else 1.0

    def decide(self, observations, force_lambda=None, force_sigma_est=None):
        """
        Decide which day to purchase skis based on observations.

        Parameters:
        observations (array-like): noisy predictions of usage days
        force_lambda (float, optional): override for the λ parameter
        force_sigma_est (float, optional): override for the estimated sigma

        Returns:
        int: the chosen day to purchase skis
        """
        # Compute average of observations
        y_bar = np.mean(observations)

        # Determine effective sigma after averaging n predictions
        if force_sigma_est is not None:
            sigma_eff = force_sigma_est / np.sqrt(self.n)
        else:
            sigma_local = np.std(observations)
            sigma_eff = sigma_local / np.sqrt(self.n)

        # Get algorithm parameters
        lambda_val, alpha, beta, b_adj = self.set_parameters(sigma_eff)
        if force_lambda is not None:
            lambda_val = force_lambda

        # If the average prediction exceeds the adjusted threshold, use the "k-branch"
        if y_bar >= b_adj:
            k = int(lambda_val * (self.b + beta * sigma_eff))
            k = max(k, 1)
            idx = np.arange(k)
            numerator = ((self.b - 1) / self.b) ** (k - idx - 1)
            denominator = self.b * (1 - (1 - 1/self.b) ** k)
            adjustment = 1 + (y_bar / self.b)
            q = numerator * adjustment / denominator
            total = q.sum()
            if total <= 0 or np.isnan(total):
                decision_day = 1  # fallback
            else:
                q /= total
                decision_day = np.random.choice(np.arange(1, k+1), p=q)

        # Otherwise, use the "ℓ-branch"
        else:
            ell = int((self.b + beta * sigma_eff) / lambda_val)
            ell = max(ell, 1)
            idx = np.arange(ell)
            numerator = ((self.b - 1) / self.b) ** (ell - idx - 1)
            denominator = self.b * (1 - (1 - 1/self.b) ** ell)
            adjustment = max(1 - (y_bar / self.b), 1e-6)
            r = numerator * adjustment / denominator
            total = r.sum()
            if total <= 0 or np.isnan(total):
                decision_day = 1  # fallback
            else:
                r /= total
                decision_day = np.random.choice(np.arange(1, ell+1), p=r)

        return decision_day
