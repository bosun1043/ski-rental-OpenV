import numpy as np

from .multiGaussian import MultiGaussianAdaptedSkiRental

class VarianceAwareSkiRental(MultiGaussianAdaptedSkiRental):
    def decide(self, observations, force_lambda=None, force_sigma_est=None):
    # Generate n predictions with noise
        y_bar = np.mean(observations)

        if force_sigma_est is not None:
            sigma_eff = force_sigma_est / np.sqrt(self.n)
        else:
            sigma_est_local = np.std(observations)
            sigma_eff = sigma_est_local / np.sqrt(self.n)

        lambda_val, alpha, beta, b_adj = self.set_parameters(sigma_eff)
        if force_lambda is not None:
            lambda_val = force_lambda

        if y_bar >= b_adj:
            # Compute k branch
            k = int(lambda_val * (self.b + beta * sigma_eff))
            if k < 1:
                k = 1
            idx = np.arange(k)
            numerator = ((self.b - 1) / self.b) ** (k - idx - 1)
            denominator = self.b * (1 - (1 - 1/self.b)**k)
            adjustment = (1 + (y_bar / self.b)) / (1 + sigma_eff)
            q = numerator * adjustment / denominator
            s = q.sum()
            if s <= 0 or np.isnan(s):
                decision_day = 1  # fallback strategy
            else:
                q /= s
                decision_day = np.random.choice(np.arange(1, k+1), p=q)
        else:
            # Compute ell branch
            ell = int((self.b + beta * sigma_eff) / lambda_val)
            if ell < 1:
                ell = 1
            idx = np.arange(ell)
            numerator = ((self.b - 1) / self.b) ** (ell - idx - 1)
            denominator = self.b * (1 - (1 - 1/self.b)**ell)
            adjustment = max(1e-6, (1 - (y_bar / self.b)) / (1 + sigma_eff))
            # Clamp adjustment to ensure it's positive
            adjustment = max(adjustment, 1e-6)
            r = numerator * adjustment / denominator
            s = r.sum()
            if s <= 0 or np.isnan(s):
                decision_day = 1  # fallback strategy
            else:
                r /= s
                decision_day = np.random.choice(np.arange(1, ell+1), p=r)

        return decision_day