/-
Copyright (c) 2025 Matteo Cipollina. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Matteo Cipollina
-/
import PhysLean.Relativity.LorentzAlgebra.Basic
import PhysLean.Relativity.LorentzGroup.Basic
import Mathlib.Analysis.Normed.Algebra.MatrixExponential
import Mathlib.Analysis.Normed.Field.Instances
import Mathlib.Topology.Metrizable.CompletelyMetrizable
/-!
# The Exponential Map from the Lorentz Algebra to the Lorentz Group

This file proves the theorem that the exponential of an element of the
Lorentz algebra is an element of the Lorentz group.

## Main results

- `exp_mem_lorentzGroup`: `A ∈ lorentzAlgebra → exp(A) ∈ LorentzGroup`.
-/

open Matrix
open minkowskiMatrix

attribute [local instance] Matrix.linftyOpNormedAlgebra
attribute [local instance] Matrix.linftyOpNormedRing
attribute [local instance] Matrix.instCompleteSpace

noncomputable section

namespace lorentzAlgebra

/--
A key property of a Lorentz algebra element `A` is that its transpose
is related to its conjugation by the Minkowski metric η.
-/
lemma transpose_eq_neg_eta_conj (A : lorentzAlgebra) :
    A.1ᵀ = - (η * A.1 * η) := by
  have h := lorentzAlgebra.transpose_eta A
  calc
    A.1ᵀ = A.1ᵀ * 1             := by rw [mul_one]
    _    = A.1ᵀ * (η * η)       := by rw [minkowskiMatrix.sq]
    _    = (A.1ᵀ * η) * η       := by rw [mul_assoc]
    _    = (-η * A.1) * η       := by rw [h]
    _    = - (η * A.1 * η)      := by simp_all only [neg_mul]

/--
The exponential of the transpose of a Lorentz algebra element.
This connects `exp(Aᵀ)` to a conjugation of `exp(-A)`.
-/
lemma exp_transpose_of_mem_algebra (A : lorentzAlgebra) :
    (NormedSpace.exp ℝ) (A.1ᵀ) = η * ((NormedSpace.exp ℝ) (-A.1)) * η := by
  rw [transpose_eq_neg_eta_conj A]
  let P_gl : GL (Fin 1 ⊕ Fin 3) ℝ :=
    { val     := η,
      inv     := η,
      val_inv := minkowskiMatrix.sq,
      inv_val := minkowskiMatrix.sq }
  rw [show -(η * A.1 * η) = η * (-A.1) * η by noncomm_ring]
  erw [NormedSpace.exp_units_conj ℝ P_gl (-A.1)]
  simp [P_gl]

/--
The exponential of an element of the Lorentz algebra is a member of the Lorentz group.
-/
theorem exp_mem_lorentzGroup (A : lorentzAlgebra) : (NormedSpace.exp ℝ) A.1 ∈ LorentzGroup 3 := by
  rw [LorentzGroup.mem_iff_transpose_mul_minkowskiMatrix_mul_self]
  rw [← Matrix.exp_transpose]
  rw [exp_transpose_of_mem_algebra A]
  calc
    (η * (NormedSpace.exp ℝ) (-A.1) * η) * η * (NormedSpace.exp ℝ) A.1
    _ = η * (NormedSpace.exp ℝ) (-A.1) * (η * η) * (NormedSpace.exp ℝ) A.1 := by noncomm_ring
    _ = η * (NormedSpace.exp ℝ) (-A.1) * 1 * (NormedSpace.exp ℝ) A.1   := by rw [minkowskiMatrix.sq]
    _ = η * (NormedSpace.exp ℝ) (-A.1 + A.1)             := by
                                            rw [mul_one, mul_assoc, NormedSpace.exp_add_of_commute]
                                            exact Commute.neg_left rfl
    _ = η * (NormedSpace.exp ℝ) 0                        := by rw [neg_add_cancel]
    _ = η * 1                            := by rw [NormedSpace.exp_zero]
    _ = η                                := by rw [mul_one]

end lorentzAlgebra
