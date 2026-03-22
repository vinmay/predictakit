## Part 6: Gradient Descent — How the Model Learns

This is the big one. Everything before this was setup. This is where the model actually learns.

Here's the situation: we have data (matches), we can make predictions (dot product with weights), and we can measure how wrong we are (MSE). Now we need to **adjust the weights to make MSE smaller.**

Imagine you're standing on a hilly landscape in complete fog. You can't see anything. But you can feel the ground under your feet — you know which direction slopes downward. Your goal is to reach the lowest point (minimum MSE).

What do you do? **You take a step downhill.** Feel the ground again. Take another step downhill. Repeat until the ground is flat (you've reached the bottom).

That's gradient descent. The "gradient" is the slope of the ground. The "descent" is stepping downhill.

Now let's make it mathematical. We need to answer: **if I nudge a weight slightly, does MSE go up or down?**

Let's think about it with a single weight. Say `weight_for_home_goals = 0.45` and our MSE is 0.0442. If we increase the weight to 0.46, maybe MSE drops to 0.0430. Good — that direction helps. If we increase to 0.47, maybe MSE drops to 0.0425. Still helping. We keep going until MSE starts going back up.

But we have 6 weights. We need to know the slope for *each weight simultaneously*. That collection of slopes is the **gradient** — it's a vector that points in the direction of steepest increase. We go the *opposite* direction.

Here's the formula for the gradient of MSE with respect to the weights. I'm going to show it, then explain every piece:
```
gradient = (2/n) * X^T · (predictions - actual)
```

Let me break that down:

- `predictions - actual` — the errors. You already computed this.
- `X^T` — the transpose of your data matrix (rows become columns). This maps errors back to features.
- `(2/n)` — the averaging factor from MSE, with the 2 from the derivative of squaring.
- The result is a vector with one value per weight — "move this weight in this direction by this much."

Then the update rule:
```
weights = weights - learning_rate * gradient