import PhysLean.StatisticalMechanics.SpinGlasses.HopfieldNetwork.TwoState.Core

open Finset Matrix NeuralNetwork State Constants Temperature Filter Topology
open scoped ENNReal NNReal BigOperators
open NeuralNetwork

--variable {R U σ : Type}
--variable {R U σ : Type*}
universe uR uU uσ

-- We can also parametrize earlier variables with these universes if desired:
variable {R : Type uR} {U : Type uU} {σ : Type uσ}
variable [DecidableEq U] [Fintype U] [Nonempty U]
namespace TwoState

/-- Visible/hidden partition for RBM-style networks. -/
structure RBMPartition (U : Type*) where
  (isVis isHid : U → Prop)
  (disjoint : ∀ u, ¬ (isVis u ∧ isHid u))
  (cover : ∀ u, isVis u ∨ isHid u)

/-- Class for (restricted) Boltzmann Machine weight constraints:
    zero diagonal and no intra-layer edges. -/
class IsRBM (NN : NeuralNetwork ℝ U σ)
    [DecidableEq U] [Fintype U] (P : RBMPartition U) : Prop where
  (no_vis_vis : ∀ {u v}, P.isVis u → P.isVis v → u ≠ v → NN.Adj u v = False)
  (no_hid_hid : ∀ {u v}, P.isHid u → P.isHid v → u ≠ v → NN.Adj u v = False)
  (symm : ∀ u v, NN.Adj u v = NN.Adj v u)

end TwoState
