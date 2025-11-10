import numpy as np
from scipy.stats import norm

class BayesianSkiRental:
    def __init__(self, b=10, n=5, prior_mu=None, prior_sigma=None):
        self.b = b
        self.n = n
        self.prior_mu = prior_mu if prior_mu is not None else b / 2
        self.prior_sigma = prior_sigma if prior_sigma is not None else b / 2
        self.posterior_mu = self.prior_mu
        self.posterior_sigma = self.prior_sigma

    def update_posterior(self, observations, observation_sigma):
        """
        Update the posterior distribution based on observations.
        """
        y_bar = np.mean(observations)

        # Set a safe observation variance
        obs_var = (observation_sigma ** 2) / self.n if observation_sigma > 0 else 1e-6

        prior_precision = 1.0 / (self.posterior_sigma ** 2)
        observation_precision = 1.0 / obs_var

        posterior_precision = prior_precision + observation_precision
        posterior_var = 1.0 / posterior_precision

        self.posterior_mu = (
            (self.posterior_mu * prior_precision) +
            (y_bar * observation_precision)
        ) / posterior_precision

        self.posterior_sigma = np.sqrt(posterior_var)
        return self.posterior_mu, self.posterior_sigma

    def decide(self, observations, force_lambda=None, force_sigma_est=None):
        """
        Decide whether to buy or rent based on the updated posterior.
        Returns 1 if buying immediately is recommended,
        otherwise returns the optimal day to switch.
        """
        sigma_est = force_sigma_est if force_sigma_est is not None else np.std(observations)
        self.update_posterior(observations, sigma_est)

        # Probability that total rental days exceed the buy cost
        p_exceed_b = 1 - norm.cdf(self.b, loc=self.posterior_mu, scale=self.posterior_sigma)
        if p_exceed_b >= 0.7:
            return 1  # Buy immediately

        # Search up to posterior_mean + 3*posterior_std (at least 30 days)
        max_day = max(int(self.posterior_mu + 3 * self.posterior_sigma), 30)
        days = np.arange(1, max_day + 1)
        expected_costs = np.zeros(len(days))

        for i, d in enumerate(days):
            buy_cost = (d - 1) + self.b
            p_up_to_d = norm.cdf(d, loc=self.posterior_mu, scale=self.posterior_sigma)

            if p_up_to_d > 0.001:
                z = (d - self.posterior_mu) / self.posterior_sigma
                # Conditional expectation of days given they are less than d
                conditional_mean = self.posterior_mu - self.posterior_sigma * (norm.pdf(z) / p_up_to_d)
            else:
                conditional_mean = 0

            cost_if_less = conditional_mean * p_up_to_d
            cost_if_more = buy_cost * (1 - p_up_to_d)
            expected_costs[i] = cost_if_less + cost_if_more

        # Choose the day with minimal expected cost
        return days[np.argmin(expected_costs)]

    def cost(self, decision_day, actual_days):
        """
        Compute the cost of the decision:
        - If you switch (buy) on or before the actual usage days, cost includes rentals plus buy price.
        - Otherwise, you just pay for rentals for the actual days.
        """
        if decision_day <= actual_days:
            return (decision_day - 1) + self.b
        else:
            return actual_days

    def competitive_ratio(self, decision_day, actual_days):
        """
        Compute the competitive ratio of the algorithm:
        algorithm cost / optimal cost (min(actual_days, b))
        """
        alg_cost = self.cost(decision_day, actual_days)
        opt_cost = min(actual_days, self.b)
        return alg_cost / opt_cost if opt_cost > 0 else 1.0
