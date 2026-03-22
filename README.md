# PredictaKit
 
A sports prediction engine where every ML algorithm is built from scratch.
 
I'm a software engineer learning machine learning by implementing it. PredictaKit predicts match outcomes using algorithms I write by hand with NumPy, and every prediction explains *why* it made that call.
 
Follow along via commits and the [learning journal](docs/learning_journal.md).
 
---
 
## The idea
 
Point it at historical match data (football, cricket, basketball), and it predicts outcomes with confidence scores and factor breakdowns. The engine (**PredictaCore**, living in `core/`) is the from-scratch ML — linear regression, logistic regression, decision trees, ensembles. The toolkit wraps it into something usable.
 
## Where I am
 
- [x] Project setup
- [ ] Linear regression from scratch
- [ ] Logistic regression from scratch
- [ ] Decision trees from scratch
- [ ] Ensemble + explanation engine
- [ ] Multi-sport features, CLI, ship it
 
## Rules I set for myself
 
- No sklearn, PyTorch, or TensorFlow in the engine. NumPy only.
- Every prediction must be explainable.
- I write every algorithm. AI helps with scaffolding and tests, not the math.
 
## License
 
MIT
