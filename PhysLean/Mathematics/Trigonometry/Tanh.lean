/-
Copyright (c) 2025 Afiq Hatta. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Afiq Hatta
-/
import PhysLean.Meta.Linters.Sorry
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Properties of Tanh
We want to prove that the reflectionless potential is a Schwartz map.
This means proving that pointwise multiplication of a Schwartz map with tanh is a Schwartz map.
This means we need to prove that all derivatives of tanh are bounded and continuous, so that
the nth derivative of a function multiplied by tanh decays faster than any polynomial.

## TODO
- Add these to mathlib eventually
- Fill in the proofs for the properties of tanh
-/

open Real
open NNReal
open Field

/-- tanh(x) is less than 1 for all x -/
lemma tanh_lt_one (x : ℝ) : tanh x < 1 := by
  rw [tanh_eq_sinh_div_cosh, div_lt_one (cosh_pos x)]
  exact sinh_lt_cosh x

/-- tanh(x) is greater than -1 for all x -/
lemma minus_one_lt_tanh (x : ℝ) : -1 < tanh x := by
  rw [tanh_eq_sinh_div_cosh, lt_div_iff₀ (cosh_pos x), ← sub_pos, neg_one_mul]
  simp [exp_pos x]

/-- The absolute value of tanh is bounded by 1 -/
lemma abs_tanh_lt_one (x : ℝ) : |tanh x| < 1 := by
  rw [abs_lt]
  constructor
  · exact minus_one_lt_tanh x
  · exact tanh_lt_one x

/-- The derivative of tanh(x) is 1 - tanh(x)^2 -/
@[sorryful]
lemma deriv_tanh (x: ℝ) : deriv Real.tanh = fun x => 1 - Real.tanh x ^ 2 := by
  sorry

/-- Tanh(x) is n times continuously differentiable for all n -/
lemma contDiff_tanh {n : ℕ} : ContDiff ℝ n tanh := by
  have hdiv : ContDiff ℝ n (fun x => Real.sinh x / Real.cosh x) := by
    apply ContDiff.div
    · exact contDiff_sinh
    · exact contDiff_cosh
    · intro x
      exact ne_of_gt (Real.cosh_pos x)
  conv =>
    enter [3, x]
    rw [tanh_eq_sinh_div_cosh]
  exact hdiv

/-- The nth derivative of Tanh(x) is a polynomial of Tanh(x) -/
@[sorryful]
lemma iteratedDeriv_tanh_is_polynomial_of_tanh (n : ℕ) (x : ℝ) : ∃ P : Polynomial ℝ, ∀ x,
    iteratedDeriv n Real.tanh x = P.eval (Real.tanh x) := by sorry
