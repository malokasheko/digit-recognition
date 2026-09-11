# Technical Research & Conceptual Analysis
### Task 3 — Handwritten Digits Recognition Neural Network, Alex Eagles Bank

All three topics (A, B, C) are covered below.

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

---

### Topic C: Weight Initialization Strategies

**The "all-zeros" trap (symmetry breaking)**

If every weight in a layer is initialized to the same constant (zero or otherwise), every
neuron in that layer computes the exact same function of the input, because they all start
with identical weights and see identical inputs. During backpropagation, the gradient with
respect to each of those neurons' weights is therefore also identical, so every neuron gets
updated in exactly the same way at every step. The layer effectively behaves as if it had only
a single neuron, no matter how many units it actually contains — the neurons never
"differentiate" from each other. This is why weights (though not necessarily biases) must be
initialized with some randomness: it breaks the symmetry so different neurons can learn
different features.

**Naive random initialization (vanishing/exploding activations)**

Randomness alone isn't sufficient, though — the *scale* of the random values matters a great
deal in deep networks. If initial weights are too small, each layer's output shrinks relative
to its input (since output variance scales with weight variance times the number of inputs
summed), so activations (and, during backprop, gradients) shrink geometrically as they pass
through many layers, eventually vanishing to near-zero and stalling learning in early layers.
If initial weights are too large, the opposite happens: activations and gradients grow
geometrically layer after layer, exploding into very large or unstable (even `NaN`) values.
Both failure modes get worse as networks get deeper, since the shrinking or growing compounds
multiplicatively across layers.

**Modern heuristics: Xavier/Glorot and He/Kaiming initialization**

The core intuition behind both schemes is the same: choose the variance of the initial weights
so that the variance of the signal is roughly preserved as it passes through a layer — neither
shrinking nor growing — in both the forward pass (activations) and the backward pass
(gradients). This means the initialization variance is chosen based on the number of input and
output connections a layer has (`fan_in` and `fan_out`), rather than picking an arbitrary fixed
scale.

Xavier/Glorot initialization derives its variance assuming a linear or near-linear activation
around zero, which is a reasonable approximation for **Tanh** (and Sigmoid) since those
functions behave close to linearly for inputs near zero and are symmetric around zero. He/Kaiming
initialization instead accounts for **ReLU**, which zeros out roughly half of its inputs (all the
negative ones) — this halves the effective variance passed forward compared to a linear
activation, so He initialization uses a larger variance (scaled by a factor of 2 relative to
Xavier's assumption) to compensate for that "lost" half. Because ReLU and its variants are the
dominant activation in modern deep networks (including the MLP built in this notebook), He
initialization is generally the recommended default whenever ReLU-family activations are used,
while Xavier remains the better fit for Tanh/Sigmoid-based networks.
