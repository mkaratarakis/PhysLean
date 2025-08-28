import PhysLean.StatisticalMechanics.SpinGlasses.HopfieldNetwork.NeuralNetwork
import Mathlib.Probability.ProbabilityMassFunction.Constructions

universe uR uU uσ

open NeuralNetwork State

namespace NeuralNetwork

/-- Probability Mass Function over Neural Network States -/
def StatePMF
  {R : Type uR} {U : Type uU} {σ : Type uσ} [Zero R]
  (NN : NeuralNetwork R U σ) : Type _ :=
  PMF NN.State

/-- Temperature-parameterized stochastic dynamics for neural networks -/
def StochasticDynamics
  {R : Type uR} {U : Type uU} {σ : Type uσ} [Zero R]
  (NN : NeuralNetwork R U σ) :=
  ℝ → NN.State → NeuralNetwork.StatePMF NN

/-- Metropolis acceptance decision as a probability mass function over Boolean outcomes -/
def State.metropolisDecision (p : ℝ) : PMF Bool :=
  PMF.bernoulli (min (Real.toNNReal p) 1) (by
    simp)

end NeuralNetwork
