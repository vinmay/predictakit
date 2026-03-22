# PredictaKit — Learning Journal

## Phase 1: Linear Regression from Scratch

**Date:** March 2026

---

### What I built

A complete linear regression model trained with gradient descent, using only NumPy for matrix operations. No sklearn, no libraries doing the thinking for me.

- `dot_product()` - element-wise multiply and sum, implemented with a plain loop first
- `matrix_vector_multiply()` - batch predictions using my dot product function
- `mse()` - Mean Squared Error loss function
- `train_linear_regression()` - the full gradient descent training loop

---

### Concepts that clicked

**Vectors are just match stats in a list.** A match like `[2, 14, 6, 1, 8, 3]` is a point in 6-dimensional space. Similar matches are close together, different matches are far apart. I could look at two match vectors and tell which was a more dominant home win before doing any math — and that intuition is exactly what the model learns to do with weights.

**The dot product is how a prediction is made.** Multiply each feature by its weight, add them up, get a single number. When I doubled the home_goals weight from 0.45 to 0.90, the prediction shifted because the model was paying more attention to that feature. That's feature importance — and I understood it before I knew the term.

**MSE punishes big errors disproportionately.** One prediction off by 4.0 (squared = 16.0) dominates the loss even if four other predictions are perfect. MSE of 3.2 from one bad match out of five. This means the model will focus on fixing its worst predictions, which is usually what you want.

**Gradient descent is "walk downhill in fog."** You can't see the whole landscape, but you can feel the slope. The gradient tells each weight which direction to move and by how much. Then you step. Repeat 1000 times and you're at the bottom.

**The bias column trick is clever.** Adding a column of 1s to the data lets the model learn a y-intercept through the same dot product — no special handling. `1 * bias_weight` just adds a constant to every prediction. Took me a minute to see why, but once it clicked it was elegant.

---

### Things that confused me (and how I resolved them)

**"Why random weights?"** I set all weights to 0.2 and wasn't sure if that was okay. Turns out for linear regression, starting weights genuinely don't matter as the loss landscape is a bowl shape (convex), so every starting point leads to the same answer. Small values are the only rule. Random initialization matters more for neural networks later because of symmetry breaking.

**"Should I have 3 weights or 4 after adding the bias column?"** Four. One per column, including the bias. The dot product needs both vectors the same length. The bias weight is just another weight — it pairs with the column of 1s.

**"Is the gradient a single value or a list?"** A list — one gradient value per weight. The transpose in `X^T · errors` maps the errors back to features, giving each weight its own adjustment direction. The shape math: `(4×5) · (5) = (4)` — four gradient values for four weights.

**"In the gradient formula, what is n?"** The number of matches (rows of data), not the number of weights. It comes from the mean in MSE — you're averaging over data points.

**"Is computing errors the same as computing MSE?"** No. Raw errors (`predictions - y`) are a vector used in the gradient calculation. MSE is the mean of squared errors — used for monitoring progress, not for the learning step itself.

---

### What I learned from the learning rate experiment

- **lr = 0.01**: smooth convergence, MSE decreased steadily
- **lr = 1.0**:  overflow, values exploded to infinity. Steps were so big the weights overshot the valley and bounced harder each time (divergence)
- **lr = 0.0001**:  still learning but painfully slow. MSE was still high after 1000 epochs. Needed way more iterations to converge

There's no formula for the right learning rate. You experiment.

---

### Code I refactored and why

**Stopped mutating input data.** My first version used `x.insert(0, 1)` to add the bias column, which modified the original data. If you call the function twice, you get two bias columns. Fixed by using `np.column_stack([np.ones(len(X)), X])` to create a new array.

**Switched from list comprehensions to NumPy operations.** I was writing things like:
```python
gradient = [scaling_factor * x for x in np.dot(X_transposed, error)]
weights = [x - (lr * y) for x, y in zip(weights, gradient)]
```
Replaced with:
```python
gradient = scaling_factor * np.dot(X_t, errors)
weights = weights - lr * gradient
```
NumPy handles element-wise operations automatically (broadcasting). Cleaner, faster, and scales to large datasets.

**Moved the transpose outside the loop.** `X` doesn't change during training, so computing `np.transpose(X_b)` every epoch was wasted work. Compute it once before the loop.

**Functions should return, not just print.** My first dot product only printed the result. Changed to return so the output could be used downstream in other calculations.

---

### The trained model's weights

```
bias:       -0.295
home_goals:  0.789
home_SOT:    0.104
away_goals: -0.954
```

Predicting goal difference, the model learned: home goals matter a lot (positive), away goals matter a lot (negative), and shots on target matter a little. Since goal difference is literally `home_goals - away_goals`, the ideal weights would be exactly 1.0 and -1.0. The model got close — 0.79 and -0.95 — with some weight leaking into SOT because goals and shots on target are correlated. With more data this would sharpen.

---

### What I'd tell someone about to learn this

Start with the dot product. Seriously. Implement it with a loop. Once you feel every multiplication and addition, everything else builds on top of it. Matrix multiplication is just batch dot products. Predictions are just dot products with weights. Gradient descent is just adjusting the weights so the dot products give better answers.

The math is not the hard part, its the WHY that is critical and hard to understand. Once you understand why you are doing a certain mathematical operation, its easier to know how you are thinking like a model thinks.