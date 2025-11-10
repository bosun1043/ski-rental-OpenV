import numpy as np

class HybridVarianceAwareSkiRental:
    def __init__(self, b=10, n=5, gamma=0.5):
        self.b = b
        self.n = n
        self.gamma = gamma

    def set_parameters(self, sigma_eff):
        lambda_val = min(1, np.sqrt(self.b / (self.b + sigma_eff))) if sigma_eff > 0 else 1
        alpha = min(1.96, np.sqrt(2 * np.log(self.b / sigma_eff))) if sigma_eff > 0 else 1.96
        beta = min(1, 2 * sigma_eff / self.b) if sigma_eff > 0 else 0
        b_adj = self.b + alpha * sigma_eff
        return lambda_val, alpha, beta, b_adj

    def get_distribution(self, y_bar, sigma_eff, force_lambda=None):
        lambda_val, alpha, beta, b_adj = self.set_parameters(sigma_eff)
        if force_lambda is not None:
            lambda_val = force_lambda

        uncertainty_factor_pos = (1 + self.gamma * y_bar / self.b) / (1 + self.gamma * sigma_eff)
        uncertainty_factor_neg = (1 - self.gamma * y_bar / self.b) / (1 + self.gamma * sigma_eff)
        uncertainty_factor_neg = max(uncertainty_factor_neg, 1e-6)

        if y_bar >= b_adj:
            k = int(lambda_val * (self.b + beta * sigma_eff))
            k = max(k, 1)
            idx = np.arange(k)
            numerator = ((self.b - 1) / self.b) ** (k - idx - 1)
            q = numerator * uncertainty_factor_pos
            s = q.sum()
            return (q / s if s > 0 and not np.isnan(s) else np.array([1.0])), 'k'
        else:
            ell = int((self.b + beta * sigma_eff) / lambda_val)
            ell = max(ell, 1)
            idx = np.arange(ell)
            numerator = ((self.b - 1) / self.b) ** (ell - idx - 1)
            r = numerator * uncertainty_factor_neg
            s = r.sum()
            return (r / s if s > 0 and not np.isnan(s) else np.array([1.0])), 'ell'

    def decide(self, observations, force_lambda=None, force_sigma_est=None):
        y_bar = np.mean(observations)
        if force_sigma_est is not None:
            sigma_eff = force_sigma_est / np.sqrt(len(observations))
        else:
            sigma_eff = np.std(observations) / np.sqrt(len(observations))

        probs, branch = self.get_distribution(y_bar, sigma_eff, force_lambda)
        return np.random.choice(np.arange(1, len(probs)+1), p=probs)

    def cost(self, decision_day, actual_days):
        if decision_day <= actual_days:
            return (decision_day - 1) + self.b
        else:
            return actual_days

    def competitive_ratio(self, decision_day, actual_days):
        alg_cost = self.cost(decision_day, actual_days)
        opt_cost = min(actual_days, self.b) if actual_days > 0 else 1.0
        return alg_cost / opt_cost if opt_cost > 0 else 1.0


class HybridVarianceAwareSkiRentalWithScaling(HybridVarianceAwareSkiRental):
    def __init__(self, b=100, n=5, gamma=0.5, c=1.0):
        super().__init__(b=b, n=n, gamma=gamma)
        self.c = c

    def set_parameters(self, sigma_eff):
        adjusted_sigma_eff = sigma_eff / self.c if self.c > 0 else sigma_eff
        return super().set_parameters(adjusted_sigma_eff)

    def decide(self, observations, force_lambda=None, force_sigma_est=None):
        y_bar = np.mean(observations)

        if force_sigma_est is not None:
            raw_sigma_eff = force_sigma_est / np.sqrt(len(observations))
        else:
            raw_sigma_eff = np.std(observations) / np.sqrt(len(observations))

        sigma_eff = raw_sigma_eff / self.c if self.c > 0 else raw_sigma_eff
        probs, branch = self.get_distribution(y_bar, sigma_eff, force_lambda)
        return np.random.choice(np.arange(1, len(probs)+1), p=probs)
