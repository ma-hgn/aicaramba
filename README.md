
# aicaramba ✨

a simple neural network implementation from a perspective of linear algebra.

the library is mostly developed for recreational and educational purposes for myself
and only depends on the `rand`-crate for randomization of newly created weight-
and bias matrices.

for a usage example see `src/bin/xor.rs`, which simulates an XOR-logic-gate using
a small neural network.

---

## features

currently available features of the library:

- ReLU and Sigmoid activation functions
- the MSE loss function
- a single, down-to-earth struct that contains the whole network


## roadmap

what might happen down the road:

- BCE loss function (requires output layer sigmoid activation - not a trivial addition)
- serde (de-)serialization to easily store checkpoints/training progress.
- perhaps a MNIST example (?)
