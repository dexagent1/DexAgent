"""Proximal policy optimization (Schulman et al. 2017) in numpy: a Gaussian policy and a value function, each a tanh MLP,
trained with the clipped surrogate objective, generalized advantage estimation and Adam. Small enough to run inside the
library's CPU environment; the policy's parameters are plain arrays, so worker processes act with a copy.

    agent = PPO(obs_dim, act_dim, hidden=64, seed=0)
    a, logp = agent.act(obs, rng)                    # one action, or the mean with deterministic=True
    agent.update(obs, act, logp, adv, ret)           # after gae(...)
    p = agent.params(); Policy(p).act(obs, rng)      # the actor alone, in another process
"""
from __future__ import annotations

import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W = [rng.normal(0.0, 1.0 / np.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.W[-1] *= out_scale
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x):
        hs = [np.asarray(x, float)]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return hs[-1], hs

    def backward(self, hs, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ d
            gb[i] = d.sum(axis=0)
            if i:
                d = (d @ self.W[i].T) * (1.0 - hs[i] ** 2)
        return gW, gb

    def arrays(self):
        return self.W + self.b


class Adam:
    def __init__(self, arrays, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros_like(a) for a in arrays]
        self.v = [np.zeros_like(a) for a in arrays]

    def step(self, arrays, grads):
        self.t += 1
        for a, g, m, v in zip(arrays, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            a -= self.lr * mh / (np.sqrt(vh) + self.eps)


class RunningNorm:
    """Observation normalization from every observation seen so far (parallel variance)."""

    def __init__(self, dim):
        self.mean, self.var, self.n = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        x = np.asarray(x, float).reshape(-1, len(self.mean))
        m, v, n = x.mean(axis=0), x.var(axis=0), len(x)
        tot = self.n + n
        delta = m - self.mean
        self.mean = self.mean + delta * n / tot
        self.var = (self.var * self.n + v * n + delta ** 2 * self.n * n / tot) / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((np.asarray(x, float) - self.mean) / np.sqrt(self.var + 1e-8), -5.0, 5.0)


class Policy:
    """The actor from params(): what a worker process needs to act."""

    def __init__(self, params):
        self.W = [np.asarray(w) for w in params["W"]]
        self.b = [np.asarray(b) for b in params["b"]]
        self.log_std = np.asarray(params["log_std"])
        self.mean_obs, self.var_obs = np.asarray(params["obs_mean"]), np.asarray(params["obs_var"])

    def norm(self, obs):
        return np.clip((np.asarray(obs, float) - self.mean_obs) / np.sqrt(self.var_obs + 1e-8), -5.0, 5.0)

    def mean(self, obs):
        h = self.norm(obs)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return h

    def act(self, obs, rng=None, deterministic=False):
        mu = self.mean(obs)
        if deterministic:
            return mu, 0.0
        std = np.exp(self.log_std)
        a = mu + std * rng.standard_normal(mu.shape)
        logp = float(-0.5 * np.sum(((a - mu) / std) ** 2 + 2 * self.log_std + LOG_2PI))
        return a, logp


def gae(rewards, values, dones, last_value, gamma=0.98, lam=0.95):
    """Advantages and returns of one trajectory; `dones[t]` true where the episode ended (no bootstrap past it)."""
    T = len(rewards)
    adv = np.zeros(T)
    run = 0.0
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[t + 1]
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        run = delta + gamma * lam * nonterminal * run
        adv[t] = run
    return adv, adv + np.asarray(values, float)


class PPO:
    def __init__(self, obs_dim, act_dim, hidden=64, lr=3e-4, seed=0, log_std=-0.7, clip=0.2, vf_coef=0.5, ent_coef=0.0,
                 max_grad_norm=0.5):
        rng = np.random.default_rng(seed)
        self.actor = MLP([obs_dim, hidden, hidden, act_dim], rng, out_scale=0.01)
        self.critic = MLP([obs_dim, hidden, hidden, 1], rng, out_scale=1.0)
        self.log_std = np.full(act_dim, float(log_std))
        self.norm = RunningNorm(obs_dim)
        self.opt_a = Adam(self.actor.arrays() + [self.log_std], lr)
        self.opt_c = Adam(self.critic.arrays(), lr)
        self.clip, self.vf_coef, self.ent_coef, self.max_grad_norm = clip, vf_coef, ent_coef, max_grad_norm

    def params(self):
        return {"W": [w.copy() for w in self.actor.W], "b": [b.copy() for b in self.actor.b], "log_std": self.log_std.copy(),
                "obs_mean": self.norm.mean.copy(), "obs_var": self.norm.var.copy()}

    def act(self, obs, rng, deterministic=False):
        return Policy(self.params()).act(obs, rng, deterministic)

    def value(self, obs, normalized=False):
        x = np.atleast_2d(np.asarray(obs, float))
        v, _ = self.critic.forward(x if normalized else self.norm(x))
        return v[:, 0]

    def _clip_grads(self, grads):
        n = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if n > self.max_grad_norm:
            grads = [g * (self.max_grad_norm / n) for g in grads]
        return grads

    def update(self, obs, act, logp_old, adv, ret, epochs=4, minibatch=256, rng=None, normalized=False):
        """One PPO update on a batch; `normalized` when `obs` were normalized with the statistics the actions were
        sampled under (keep the normalizer's update for after). Returns diagnostics."""
        rng = rng or np.random.default_rng(0)
        obs, act = np.asarray(obs, float), np.asarray(act, float)
        logp_old, ret = np.asarray(logp_old, float), np.asarray(ret, float)
        adv = np.asarray(adv, float)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        x_all = obs if normalized else self.norm(obs)
        N = len(obs)
        stats = {"clip_frac": [], "approx_kl": [], "value_loss": []}
        for _ in range(int(epochs)):
            order = rng.permutation(N)
            for s in range(0, N, int(minibatch)):
                idx = order[s:s + int(minibatch)]
                x, a, lo, A, R = x_all[idx], act[idx], logp_old[idx], adv[idx], ret[idx]
                n = len(idx)
                mu, hs = self.actor.forward(x)
                std = np.exp(self.log_std)
                z = (a - mu) / std
                logp = -0.5 * np.sum(z ** 2 + 2 * self.log_std + LOG_2PI, axis=1)
                ratio = np.exp(np.clip(logp - lo, -20, 20))
                unclipped = ratio * A
                clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * A
                active = (unclipped <= clipped).astype(float)      # the minimum is the unclipped term
                dlogp = -(A * ratio * active) / n                   # d(-surrogate)/d logp
                dmu = (dlogp[:, None] * z / std)
                dlog_std = (dlogp[:, None] * (z ** 2 - 1.0)).sum(axis=0) - self.ent_coef
                gW, gb = self.actor.backward(hs, dmu)
                grads = self._clip_grads(gW + gb + [dlog_std])
                self.opt_a.step(self.actor.arrays() + [self.log_std], grads)
                np.clip(self.log_std, -3.0, 0.5, out=self.log_std)
                v, hc = self.critic.forward(x)
                dv = self.vf_coef * 2.0 * (v[:, 0] - R)[:, None] / n
                cW, cb = self.critic.backward(hc, dv)
                self.opt_c.step(self.critic.arrays(), self._clip_grads(cW + cb))
                stats["clip_frac"].append(float(np.mean(np.abs(ratio - 1) > self.clip)))
                stats["approx_kl"].append(float(np.mean(lo - logp)))
                stats["value_loss"].append(float(np.mean((v[:, 0] - R) ** 2)))
        return {k: round(float(np.mean(v)), 5) for k, v in stats.items()}
