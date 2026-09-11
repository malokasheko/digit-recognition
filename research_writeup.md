# Technical Research & Conceptual Analysis
### Task 3 — Handwritten Digits Recognition Neural Network, Alex Eagles Bank

All three topics (A and B) are covered below.

### Topic A: Escaping Traps (Local Minima vs. Saddle Points)

**Why true local minima are rare in high dimensions**

A point is a local minimum only if *every* direction you could perturb the weights in leads
uphill. In a 2D loss surface there are only two directions to check, so a "bowl" trapping the
optimizer is plausible. A modern MLP has thousands to millions of parameters, meaning a
critical point (where the gradient is zero) lives in a space with that many independent
directions. For a critical point to be a true local minimum, the loss must curve *upward*
along **all** of those directions simultaneously — equivalently, the Hessian (the matrix of
second derivatives) must be positive-definite in every eigen-direction at once. Random matrix
theory suggests that as dimensionality grows, the odds of a critical point having all-positive
curvature shrink rapidly, while the odds of a "mixed" signature — some directions curving up,
some curving down — grow. That mixed-signature critical point is exactly a **saddle point**.
So in a high-dimensional network, a critical point is overwhelmingly more likely to be a saddle
than a genuine local minimum.

**Saddle points and plateaus as the real bottleneck**

At a saddle point the gradient is zero (so vanilla gradient descent slows to a crawl nearby),
but there's always at least one descending direction available — the optimizer isn't
permanently stuck, it just needs enough noise or curvature information to find that direction
and slide off. Flat plateaus are related but distinct: broad, nearly-zero-gradient regions
where the loss doesn't have a clean escape direction nearby, just a very shallow slope. In
practice, these saddle regions and plateaus are what cause training to visibly stall for many
steps — not because the network is caught in a bad final answer, but because the gradient
signal near a saddle is too weak to make fast progress.

**How mini-batch noise helps**

Full-batch gradient descent computes the *exact* gradient of the loss averaged over the whole
dataset, so at a saddle point that exact gradient really is (close to) zero and progress
genuinely stalls. Mini-batch SGD instead computes a noisy estimate of the gradient from a small
random subset of the data each step. That noise means the gradient computed at any given step
is very unlikely to be exactly zero even when the *true* full-dataset gradient is — the
stochastic estimate effectively perturbs the optimizer off the saddle in some random direction.
Once nudged even slightly off a saddle, the descending directions that were available all along
start pulling the weights downhill again. The same noise helps on flat plateaus: instead of
following one deterministic (and possibly very slow) path, the optimizer bounces around
slightly, increasing the chance it stumbles onto a direction with a steeper drop. This is part
of why SGD-style training, despite being "noisier" than full-batch gradient descent, often
converges to good solutions faster in practice on large networks.

---

### Topic B: Comparative Analysis of Optimizers

**Stochastic Gradient Descent (SGD)**

Standard SGD updates each weight by moving a fixed step (the learning rate) in the direction
opposite the estimated gradient: `w ← w - lr * grad`. Every parameter shares the same learning
rate, and the update only ever looks at the *current* gradient with no memory of past updates.
This causes two well-known problems. First, in narrow, steep ravines — common in loss surfaces
where curvature is very different across directions — the gradient points mostly toward the
steep walls rather than along the gentle direction that actually leads to the minimum, so SGD
zig-zags back and forth across the ravine instead of moving efficiently along it. Second, in
very flat regions the gradient is tiny, so with no memory of prior momentum, progress crawls.

**SGD with Momentum**

Momentum addresses this with a simple physical analogy: imagine a heavy ball rolling downhill
instead of a weightless point that reacts only to the instantaneous slope. The update keeps a
running "velocity" that accumulates a fraction of previous gradients: `v ← β*v + grad`, then
`w ← w - lr*v`. Because the ball has inertia, oscillations across a narrow ravine — which point
in opposite directions on alternating steps — tend to cancel out in the accumulated velocity,
while the consistent downhill component (along the ravine's length) keeps reinforcing itself
and grows. The same inertia carries the ball through flat plateaus and shallow saddle regions
where the instantaneous gradient alone would barely move it.

**Adaptive Optimizers (RMSprop & Adam)**

"Adapting the learning rate per parameter" means each individual weight gets its own effective
step size, scaled by how large that weight's gradients have typically been. RMSprop tracks a
running average of the squared gradient for each parameter and divides the update by its square
root, so parameters with a history of large, volatile gradients get smaller effective steps
(damping instability), while parameters with small, consistent gradients get relatively larger
steps (speeding up otherwise-slow directions). Adam combines this per-parameter adaptive
scaling (like RMSprop) with momentum (like SGD+Momentum), tracking both a running mean and a
running variance of the gradients.

Adam is typically the default choice for quick baseline experimentation because it is
comparatively insensitive to the initial learning rate choice and tends to converge quickly
with minimal tuning — useful when you just want a working baseline fast. However, well-tuned
SGD with Momentum can still outperform Adam on final generalization in some benchmark settings
(this shows up often in image classification with CNNs, for example) — Adam's aggressive
per-parameter adaptivity can sometimes settle into sharper minima that generalize slightly
worse, whereas plain SGD with a carefully tuned learning-rate schedule and momentum can find
flatter minima that generalize better, at the cost of needing more manual tuning effort.

