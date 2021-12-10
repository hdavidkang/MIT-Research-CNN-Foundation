# How do Deep Networks avoid the Curse of Dimensionality?

Hi there 🤠

It you're into machine learning, you'd probably know that both shallow and deep networks are _universal_, but it is the family of deep networks that avoids the _curse of dimensionality_. But wait, what is the guarantee? What is the key reason for such blessing?

## 🚀 Study outline
In this study, we borrow the theoretical constructs from Professor Poggio at MIT and show that the exponential reduction of network complexity is permitted via _hierarchical locality_, not _weight sharing_. In particular, we run a set of experiments on CIFAR-10 and discuss conditions for locality's triumph.

## 🎓 Background
####  _Curse of dimensionality_
Let σ : R → R be infinitely differentiable and not a polynomial. For f ∈ W<sub>m</sub><sup>n</sup>, the complexity of shallow networks that guarantee accuracy at least ε is,

<p align="center">
  N = O(ε <sup>-m/n</sup> ) and is the best possible
</p>

Here, W<sub>m</sub><sup>n</sup> is the set of all functions of n variables with continuous partial derivatives of orders up to m < ∞. The exponential dependence on the dimension n of the number ε <sup>-m/n</sup> is known as the curse of dimensionality. Note that the degree of approximation is defined by dist(f, V<sub>N</sub>)=inf<sub>P ∈ V<sub>N</sub> </sub> ||f−P||.
For example, if dist(f, V<sub>N</sub>) = O(N−γ) for some γ > 0,the nanetwork with complexity N = O(ε <sup>1/γ</sup> ) is sufficient to guarantee an approximation with accuracy at least ε.

####  _Blessing of compositionality_
Following the argument from Poggio, deep networks avoid the curse of dimensionality, since it only requires network complexity of O((n−1)ε−<sup>2/m</sup> )to provide approximation with accuracy at least ε. Note that such exponential benefit is due to compositionality; functions composed of hierarchically local functions. That is, deep networks avoids the curse of dimensionality because of the blessing of compositionality via small effective dimension.

####  _Role of weight sharing_
One should note that weight sharing (a.k.a. invariance) also helps reduce the network complexity, but not exponentially as hierarchical locality does. In this study, we attempt to empirically show that while weight sharing helps, hierarchical locality is the main player.

## 💻 Compute resources
NVIDIA’s Tesla P100 data center GPU and a variety of AWS EC2 instances, including the M4, C5, R5, and G4 on Amazon SageMaker was used to accelerate the learning process.

## ✔️ Summary
We report that,
- _The dominant effect of hierarchical locality of kernels over weight sharing does exist._
- _Its exponential benefit is pronounced when SGD optimizer is used._
- _The effect is clear in the case of 500k number of parameters, more so than 50k._

## 🎖 Acknowledgement
This project would not have been possible if without the great guidance and beautiful lectures of Professor Tomaso Poggio and helpful tips from Akshay Rangamani, MIT 9.520 senior staff.

--------

A teaser for those interested is shown below. 

<p align="center">
  <img src="https://github.com/hdavidkang/MIT-Project-Hierarchical-Locality-v.s.-Weight-Sharing/blob/0744b70a56eafffd109ab949d73a7b653abecc1f/figures/Figure%201.%20mse-loss_500k.jpg" />
</p>
