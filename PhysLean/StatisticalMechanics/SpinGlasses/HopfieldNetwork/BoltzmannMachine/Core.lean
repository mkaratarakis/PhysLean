/-
Copyright (c) 2025 Matteo Cipollina. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Matteo Cipollina
-/

import PhysLean.StatisticalMechanics.SpinGlasses.HopfieldNetwork.DetailedBalanceBM
import PhysLean.StatisticalMechanics.SpinGlasses.HopfieldNetwork.ZeroTemp
import PhysLean.StatisticalMechanics.SpinGlasses.HopfieldNetwork.Core
import PhysLean.StatisticalMechanics.SpinGlasses.HopfieldNetwork.CoreBridgeLOF
import Mathlib.Algebra.Lie.OfAssociative
import Mathlib.Data.Real.StarOrdered
import Mathlib.Order.CompletePartialOrder
import Mathlib.Probability.Distributions.Uniform


/-!
# Boltzmann Machine Neural Network

This file defines Boltzmann Machines (BMs) as a specific instance of a Hopfield-style
neural network. A Boltzmann machine is a stochastic recurrent neural network with symmetric
weights and no self-connections, whose dynamics are governed by an energy function.

This file formalizes the Boltzmann Machine as a type synonym for `TwoState.SymmetricBinary`,
inheriting its structure and properties. This approach connects it directly to the existing
frameworks for two-state networks and canonical ensembles.

We define `ParamsBM` to hold the network parameters, including temperature, and establish
that the standard Hopfield Hamiltonian provides the correct energy specification.

## Mathematical Model

- **Network Structure**: Fully connected, symmetric weights ($w_{uv} = w_{vu}$), no self-loops ($w_{uu}=0$).
- **Activations**: Binary neurons with states $s_u \in \{+1, -1\}$.
- **Energy Function (Hamiltonian)**: $E(s) = -\frac{1}{2}\sum_{u \neq v} w_{uv}s_u s_v - \sum_u \theta_u s_u$.
- **Stochastic Dynamics**: State transitions follow a Gibbs distribution, $P(s) \propto \exp(-E(s)/T)$, where $T$ is the temperature. This is implemented via Gibbs sampling.

By defining `BoltzmannMachine` as `SymmetricBinary`, we inherit:
- The `TwoStateNeuralNetwork` instance, confirming its binary nature.
- The `EnergySpec'` from `HopfieldEnergy.symmetricBinaryEnergySpec`, which correctly links the Hamiltonian to the network's local dynamics.
- The `IsHamiltonian` instance via `IsHamiltonian_of_EnergySpec'`, bridging the network to the `CanonicalEnsemble` framework.

## Mathematics

Boltzmann Machines have binary neurons (±1) with probability of activation determined by:
- Energy function: $E(s) = -\frac{1}{2}\sum_{u,v, u \neq v} w_{u,v}s_u s_v - \sum_u \theta_u s_u$
- Probability distribution: $P(s) \propto \exp(-E(s)/T)$ where $T$ is the temperature parameter
- Local field for neuron $u$: $L_u(s) = \sum_{v \neq u} w_{u,v}s_v + \theta_u$
- Probability of neuron $u$ being 1: $P(s_u = 1) = \frac{1}{1 + \exp(-2L_u(s)/T)}$

Key derived properties for Boltzmann Machine (BM) = SymmetricBinary:

Structural
- abbrev BoltzmannMachine R U = TwoState.SymmetricBinary R U
- StateBM ≃ functions U → {+1, -1} (finite; Fintype instance via BinarySetReal)
- ParamsBM wraps core Params + temperature T > 0

Energy / Local Field
- HopfieldEnergy.hamiltonian p s = −(1/2) * sᵀ W s + θ · s
- symmetricBinaryEnergySpec : EnergySpec' (gives E, localField, flip relation)
- localField_spec: spec.localField p s u = s.net p u − θ_u
- hamiltonian_flip_relation: E(s⁺) − E(s⁻) = −2 * (net − θ)

Lyapunov / Convergence
- Instance IsStrictlyHamiltonian_of_TwoState_EnergySpec:
  * energy_is_lyapunov: E after single-site update ≤ before
  * aux_strictly_decreases_on_tie: tie broken by magnetization rank
- convergence_of_hamiltonian: fair async updates reach stable state (∃N stable)

Probabilistic Dynamics
- probPos logistic form; 0 < probPos < 1; symmetry logisticProb(-x) = 1 - logisticProb x
- gibbsUpdate / randomScanKernel defined; random-scan mixture over sites
- Zero-temperature limit: gibbs_update_tends_to_zero_temp_limit (pointwise PMF convergence)
- Explicit limiting kernel zeroTempLimitPMF (deterministic except ties → 1/2 split)

Detailed Balance & Boltzmann Distribution
- CEparams builds CanonicalEnsemble from EnergySpec'
- P p T s = Boltzmann weight / Z
- boltzmann_ratio: P(s') / P(s) = exp(−β(E(s')−E(s)))
- randomScanKernel_reversible: reversibility (detailed balance) w.r.t. Boltzmann measure

Stochastic Matrix (Random-Scan)
- RScol: column-stochastic matrix of random-scan Gibbs kernel
  * RScol_nonneg: entries ≥ 0
  * RScol_colsum_one: column sums = 1
  * RScol_diag_pos: aperiodicity (positive self-loop)
  * DiffOnly / diffSites API for Hamming distance
  * exists_single_flip_reduce: single-site flip reduces distance
  * RScol_exists_positive_power: communication (∃ n, (RScol^n) s' s > 0)
  * RScol_irred: irreducible (Perron–Frobenius strong connectivity)
  * exists_positive_eigenvector_of_irreducible_stochastic: unique stationary vector in simplex
  * randomScan_ergodicUniqueInvariant: reversibility ∧ positive diagonal ∧ irreducible ∧ unique stationary

Zero-Temperature Asymptotics
- tendsto_probPos_at_zero / scaled logistic lemmas: classification (→1 / →0 / →1/2)
- Full pointwise convergence for every state (gibbs_update_tends_to_zero_temp_limit)

Summary: BM inherits a certified Lyapunov structure, convergence of asynchronous dynamics, detailed balance, zero-temperature limit behavior, and Perron–Frobenius spectral uniqueness of the stationary distribution for the random-scan Gibbs sampler.

-/

open Finset Matrix NeuralNetwork State ENNReal Real PMF TwoState

variable {R U : Type} [Field R] [LinearOrder R] [IsStrictOrderedRing R]
  [DecidableEq U] [Fintype U] [Nonempty U]

/--
A `BoltzmannMachine` is a stochastic Hopfield network with symmetric weights and no self-loops.
We formalize it as a type synonym for `TwoState.SymmetricBinary`, inheriting its entire structure
and associated proofs, such as its `TwoStateNeuralNetwork` and `IsHamiltonian` instances.
-/
abbrev BoltzmannMachine (R U : Type) [Field R] [LinearOrder R] [IsStrictOrderedRing R]
  [DecidableEq U] [Fintype U] [Nonempty U] : NeuralNetwork R U R :=
  TwoState.SymmetricBinary R U

/--
Parameters for a Boltzmann Machine, including the temperature `T`.
This is a convenience wrapper around the parameters for a `SymmetricBinary` network.
-/
structure ParamsBM (R U : Type) [Field R] [LinearOrder R] [IsStrictOrderedRing R]
    [DecidableEq U] [Fintype U] [Nonempty U] where
  /-- The underlying parameters (weights `w` and thresholds `θ`) of the network. -/
  params : Params (BoltzmannMachine R U)
  /-- The temperature parameter of the Boltzmann Machine. -/
  T : R
  /-- Proof that the temperature `T` is positive. -/
  hT_pos : T > 0

/-- The state of a Boltzmann Machine is the state of the underlying `SymmetricBinary` network. -/
abbrev StateBM (R U : Type) [Field R] [LinearOrder R] [IsStrictOrderedRing R]
  [DecidableEq U] [Fintype U] [Nonempty U] :=
  State (BoltzmannMachine R U)

/--
The energy of a Boltzmann Machine state is given by the standard Hopfield Hamiltonian,
provided by `HopfieldEnergy.symmetricBinaryEnergySpec`.
-/
noncomputable def energy (p : ParamsBM ℝ U) (s : StateBM ℝ U) : ℝ :=
  HopfieldEnergy.hamiltonian p.params s

/--
The local field for a neuron in a Boltzmann Machine.
-/
noncomputable def localField (p : ParamsBM ℝ U) (s : StateBM ℝ U) (u : U) : ℝ :=
  HopfieldEnergy.symmetricBinaryEnergySpec.localField p.params s u

/--
The probability that a given neuron activates (is 1) under the Gibbs distribution.
This uses the `probPos` function defined for all two-state networks.
-/
noncomputable def probNeuronIsOne (p : ParamsBM ℝ U) (s : StateBM ℝ U) (u : U) : ℝ :=
  probPos (f := RingHom.id ℝ) p.params { val := ⟨p.T, le_of_lt p.hT_pos⟩ } s u

/--
A single step of Gibbs sampling for a Boltzmann Machine.
This updates a single neuron `u` according to the conditional probability derived from the energy function.
-/
noncomputable def gibbsSamplingStep (p : ParamsBM ℝ U) (s : StateBM ℝ U) (u : U) : PMF (StateBM ℝ U) :=
  gibbsUpdate (f := RingHom.id ℝ) p.params { val := ⟨p.T, le_of_lt p.hT_pos⟩ } s u
