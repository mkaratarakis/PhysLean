import PhysLean.StatisticalMechanics.SpinGlasses.Mathematics.LinearAlgebra.Matrix.PerronFrobenius.Lemmas
import PhysLean.StatisticalMechanics.SpinGlasses.Mathematics.aux
import Mathlib.Data.Matrix.Basic
namespace Matrix
open Finset Quiver Matrix
variable {n : Type*} [Fintype n]

/-!
### TThe Collatz-Wielandt function for Matrices

-/
section PerronFrobenius
variable {n : Type*} [Fintype n] [Nonempty n]
variable {A : Matrix n n ℝ}
open LinearMap Set Filter Topology Finset
open scoped Convex Pointwise

/-- The Collatz-Wielandt function, `r(x)` in Seneta's notation.
For a non-zero, non-negative vector `x`, this is `min_{i | xᵢ > 0} (Ax)ᵢ / xᵢ`.
Seneta uses row vectors `x'T`; we use column vectors `Ax`. The logic is identical. -/
noncomputable def collatzWielandtFn (A : Matrix n n ℝ) (x : n → ℝ) : ℝ :=
  let supp := {i | 0 < x i}.toFinset
  if h : supp.Nonempty then
    supp.inf' h fun i => (A *ᵥ x) i / x i
  else 0

/-
/-- The Collatz-Wielandt function r_x for a positive vector `x`.
    See p. 30 in Berman & Plemmons.
    We define it for strictly positive vectors to avoid division by zero. -/
noncomputable def r_x (A : Matrix n n ℝ) (x : n → ℝ) : ℝ :=
  ⨅ i, (A.mulVec x i) / (x i)-/

instance : Nonempty n := inferInstance

/-- The matrix-vector multiplication map as a continuous linear map. -/
noncomputable abbrev mulVec_continuousLinearMap (A : Matrix n n ℝ) : (n → ℝ) →L[ℝ] (n → ℝ) :=
  (Matrix.mulVecLin A).toContinuousLinearMap

omit [Nonempty n] in
/-- The standard simplex in ℝⁿ is compact (Heine-Borel: closed and bounded in ℝⁿ).
    [Giaquinta-Modica, Theorem 6.3, cite: 230] -/
private lemma IsCompact_stdSimplex : IsCompact (stdSimplex ℝ n) := by
  -- stdSimplex is a closed and bounded subset of ℝⁿ
  exact _root_.isCompact_stdSimplex n

namespace CollatzWielandt

/-
-- The Collatz-Wielandt function r_x is continuous on the set of strictly positive vectors.
lemma r_x_continuousOn_pos (A : Matrix n n ℝ) :
    ContinuousOn (r_x A) {x : n → ℝ | ∀ i, 0 < x i} := by
  unfold r_x
  apply continuousOn_iInf
  intro i
  apply ContinuousOn.div
  · exact ((continuous_apply i).comp (mulVec_continuousLinearMap A).continuous).continuousOn
  · exact (continuous_apply i).continuousOn
  · exact fun x a ↦ Ne.symm (ne_of_lt (a i))-/

/-- *The Collatz-Wielandt function is upper-semicontinuous*.
Seneta relies on this fact (p.15, Appendix C) to use the Extreme Value Theorem.
`r(x)` is a minimum of functions `fᵢ(x) = (Ax)ᵢ / xᵢ`. Each `fᵢ` is continuous where `xᵢ > 0`.
The minimum of continuous functions is upper-semicontinuous.
[Giaquinta-Modica, Definition 6.21, Exercise 6.28, pp: 235, 236] -/
theorem upperSemicontinuousOn
    (A : Matrix n n ℝ) : UpperSemicontinuousOn (collatzWielandtFn A) (stdSimplex ℝ n) := by
    have support_nonempty : ∀ x ∈ stdSimplex ℝ n, ({i | 0 < x i}.toFinset).Nonempty := by
      intro x hx
      obtain ⟨i, hi_pos⟩ := exists_pos_of_sum_one_of_nonneg hx.2 hx.1
      have hi_mem : i ∈ ({i | 0 < x i}.toFinset) := by simp [hi_pos]
      exact ⟨i, hi_mem⟩
    intro x₀ hx₀ c hc
    have supp_x₀ : {i | 0 < x₀ i}.toFinset.Nonempty := support_nonempty x₀ hx₀
    have fn_eq : collatzWielandtFn A x₀ = {i | 0 < x₀ i}.toFinset.inf' supp_x₀ (fun i => (A *ᵥ x₀) i / x₀ i) := by
      unfold collatzWielandtFn
      rw [dif_pos supp_x₀]
    let U := {y : n → ℝ | ∀ i ∈ {i | 0 < x₀ i}.toFinset, 0 < y i}
    have x₀_in_U : x₀ ∈ U := by
      intro i hi
      simp only [Set.mem_toFinset] at hi
      exact hi
    have U_open : IsOpen U := by
      have h_eq : U = ⋂ i ∈ {i | 0 < x₀ i}.toFinset, {y | 0 < y i} := by
        ext y
        simp only [Set.mem_iInter, Set.mem_setOf_eq]
        rfl
      rw [h_eq]
      apply isOpen_biInter_finset
      intro i _
      exact isOpen_lt continuous_const (continuous_apply i)
    let f := fun y => {i | 0 < x₀ i}.toFinset.inf' supp_x₀ fun i => (A *ᵥ y) i / y i
    have f_cont : ContinuousOn f U := by
      apply continuousOn_finset_inf' supp_x₀
      intro i hi
      apply ContinuousOn.div
      · exact continuous_apply i |>.comp_continuousOn ((mulVec_continuousLinearMap A).continuous.continuousOn)
      · exact (continuous_apply i).continuousOn
      · intro y hy
        simp only at hy
        exact (hy i hi).ne'
    have f_ge : ∀ y ∈ U ∩ stdSimplex ℝ n, collatzWielandtFn A y ≤ f y := by
      intro y hy
      have y_supp : {i | 0 < y i}.toFinset.Nonempty := support_nonempty y hy.2
      have fn_y : collatzWielandtFn A y = {i | 0 < y i}.toFinset.inf' y_supp fun i => (A *ᵥ y) i / y i := by
        unfold collatzWielandtFn
        rw [dif_pos y_supp]
      have supp_subset : {i | 0 < x₀ i}.toFinset ⊆ {i | 0 < y i}.toFinset := by
        intro i hi
        have hi' : 0 < x₀ i := by
          simp only [Set.mem_toFinset] at hi
          exact hi
        have : 0 < y i := hy.1 i hi
        simp only [Set.mem_toFinset]
        exact this
      rw [fn_y]
      apply finset_inf'_mono_subset supp_subset
    rw [fn_eq] at hc
    have : f x₀ < c := hc
    have cont_at : ContinuousAt f x₀ := f_cont.continuousAt (IsOpen.mem_nhds U_open x₀_in_U)
    have lt_eventually : ∀ᶠ y in 𝓝 x₀, f y < c :=
      Filter.Tendsto.eventually_lt_const hc cont_at
    rcases eventually_to_open lt_eventually with ⟨V, V_open, x₀_in_V, hV⟩
    let W := V ∩ U ∩ stdSimplex ℝ n
    have VU_open : IsOpen (V ∩ U) := IsOpen.inter V_open U_open
    have VU_mem : x₀ ∈ V ∩ U := ⟨x₀_in_V, x₀_in_U⟩
    have VU_nhds : V ∩ U ∈ 𝓝 x₀ := VU_open.mem_nhds VU_mem
    have W_nhdsWithin : W ∈ 𝓝[stdSimplex ℝ n] x₀ := by
      rw [mem_nhdsWithin_iff_exists_mem_nhds_inter]
      exact ⟨V ∩ U, VU_nhds, by simp [W]⟩
    have h_final : ∀ y ∈ W, collatzWielandtFn A y < c := by
      intro y hy
      apply lt_of_le_of_lt
      · apply f_ge y
        exact ⟨And.right (And.left hy), hy.2⟩
      · exact hV y (And.left (And.left hy))
    exact Filter.mem_of_superset W_nhdsWithin (fun y hy => h_final y hy)

-- The set of vectors we are optimizing over.
def P_set := {x : n → ℝ | (∀ i, 0 ≤ x i) ∧ x ≠ 0}

/-- The Collatz-Wielandt value is the supremum of `r_x` over P. -/
noncomputable def r (A : Matrix n n ℝ) [Fintype n] := ⨆ x ∈ P_set, collatzWielandtFn A x

/-- The Collatz-Wielandt function attains its maximum on the standard simplex.
    [Giaquinta-Modica, Theorem 6.24 (dual), p: 235] -/
theorem exists_maximizer (A : Matrix n n ℝ) :
    ∃ v ∈ stdSimplex ℝ n, IsMaxOn (collatzWielandtFn A) (stdSimplex ℝ n) v := by
  have h_compact : IsCompact (stdSimplex ℝ n) := by exact _root_.isCompact_stdSimplex n
  have h_nonempty : (stdSimplex ℝ n).Nonempty := stdSimplex_nonempty
  have h_usc : UpperSemicontinuousOn (collatzWielandtFn A) (stdSimplex ℝ n) :=
    upperSemicontinuousOn A
  exact IsCompact.exists_isMaxOn_of_upperSemicontinuousOn h_compact h_nonempty h_usc

lemma eq_iInf_of_nonempty
  {n : Type*} [Fintype n] [Nonempty n] (A : Matrix n n ℝ)
  (v : n → ℝ) (h : {i | 0 < v i}.toFinset.Nonempty) :
  collatzWielandtFn A v =
    ⨅ i : {i | 0 < v i}, (A *ᵥ v) i / v i := by
  dsimp [collatzWielandtFn]
  rw [dif_pos h]
  rw [Finset.inf'_eq_ciInf h]
  refine Function.Surjective.iInf_congr ?_ (fun b ↦ ?_) ?_
  intro i
  · simp_all only [toFinset_setOf]
    obtain ⟨val, property⟩ := i
    simp_all only [toFinset_setOf, mem_filter, Finset.mem_univ, true_and]
    apply Subtype.mk
    · exact property
  · simp_all
    obtain ⟨val, property⟩ := b
    simp_all only [Subtype.mk.injEq, exists_prop, exists_eq_right]
  simp

omit [Nonempty n] in
/-- If r ≤ 0 and r is the infimum of non-negative ratios, then r = 0. -/
lemma val_eq_zero_of_nonpos [DecidableEq n]
    (hA_nonneg : ∀ i j, 0 ≤ A i j) {v : n → ℝ} (hv_nonneg : ∀ i, 0 ≤ v i)
    (S : Set n) (hS_def : S = {i | 0 < v i}) (hS_nonempty : S.Nonempty)
    (r : ℝ) (hr_def : r = collatzWielandtFn A v) (hr_nonpos : r ≤ 0) :
    r = 0 := by
  have r_ge_zero : 0 ≤ r := by
    rw [hr_def, collatzWielandtFn]
    have hS_finset_nonempty : { j | 0 < v j }.toFinset.Nonempty := by
      rwa [Set.toFinset_nonempty_iff, ← hS_def]
    rw [dif_pos hS_finset_nonempty]
    apply Finset.le_inf'
    intro j hj
    rw [Set.mem_toFinset] at hj
    exact div_nonneg (Finset.sum_nonneg fun k _ => mul_nonneg (hA_nonneg j k) (hv_nonneg k)) hj.le
  exact le_antisymm hr_nonpos r_ge_zero

omit [Nonempty n] in
/-- Each ratio is at least the Collatz-Wielandt value -/
lemma le_ratio [DecidableEq n]
    (_ : ∀ i j, 0 ≤ A i j) {v : n → ℝ} (_ : ∀ i, 0 ≤ v i)
    (S : Set n) (hS_def : S = {i | 0 < v i}) (hS_nonempty : S.Nonempty)
    (i : n) (hi_S : i ∈ S) : collatzWielandtFn A v ≤ (A *ᵥ v) i / v i := by
  rw [collatzWielandtFn]
  have hS_finset_nonempty : { j | 0 < v j }.toFinset.Nonempty := by
    rwa [Set.toFinset_nonempty_iff, ← hS_def]
  rw [dif_pos hS_finset_nonempty]
  apply Finset.inf'_le
  rw [Set.mem_toFinset, ← hS_def]
  exact hi_S

omit [Nonempty n] in
/-- For any non-negative, non-zero vector `v`, the Collatz-Wielandt value `r` satisfies
    `r • v ≤ A *ᵥ v`. This is the fundamental inequality derived from the definition of
    the Collatz-Wielandt function. -/
lemma le_mulVec [DecidableEq n]
    (hA_nonneg : ∀ i j, 0 ≤ A i j) {v : n → ℝ} (hv_nonneg : ∀ i, 0 ≤ v i)
    (hv_ne_zero : v ≠ 0) :
    (collatzWielandtFn A v) • v ≤ A *ᵥ v := by
  let r := collatzWielandtFn A v
  have hS_nonempty : ({i | 0 < v i}).Nonempty :=
    exists_pos_of_ne_zero hv_nonneg hv_ne_zero
  intro i
  by_cases hi_supp : v i > 0
  · have h_le_div : r ≤ (A *ᵥ v) i / v i :=
      le_ratio hA_nonneg hv_nonneg {i | 0 < v i} rfl hS_nonempty i hi_supp
    simp only [Pi.smul_apply, smul_eq_mul]
    exact (le_div_iff₀ hi_supp).mp h_le_div
  · have h_vi_zero : v i = 0 := by linarith [hv_nonneg i, not_lt.mp hi_supp]
    simp only [Pi.smul_apply, smul_eq_mul, h_vi_zero, mul_zero]
    exact mulVec_nonneg hA_nonneg hv_nonneg i

omit [Fintype n] [Nonempty n] in
/-- If the Collatz-Wielandt value `r` is non-positive, there must be some `i` in the support of `v`
    where the ratio, and thus `(A * v) i`, is zero. -/
lemma exists_mulVec_eq_zero_on_support_of_nonpos [Fintype n]
  (hA_nonneg : ∀ i j, 0 ≤ A i j) {v : n → ℝ} (hv_nonneg : ∀ i, 0 ≤ v i)
  (h_supp_nonempty : {i | 0 < v i}.toFinset.Nonempty)
  (h_r_nonpos : collatzWielandtFn A v ≤ 0) :
  ∃ i ∈ {i | 0 < v i}.toFinset, (A *ᵥ v) i = 0 := by
  have r_nonneg : 0 ≤ collatzWielandtFn A v := by
    rw [collatzWielandtFn, dif_pos h_supp_nonempty]
    apply Finset.le_inf'
    intro i hi_mem
    exact div_nonneg (mulVec_nonneg hA_nonneg hv_nonneg i) (by exact hv_nonneg i)
  have r_eq_zero : collatzWielandtFn A v = 0 := le_antisymm h_r_nonpos r_nonneg
  rw [collatzWielandtFn, dif_pos h_supp_nonempty] at r_eq_zero
  obtain ⟨b, hb_mem, hb_eq⟩ := Finset.exists_mem_eq_inf' h_supp_nonempty (fun i => (A *ᵥ v) i / v i)
  have h_ratio_zero : (A *ᵥ v) b / v b = 0 := by rw [← hb_eq, r_eq_zero]
  have h_vb_pos : 0 < v b := by simpa [Set.mem_toFinset] using hb_mem
  exact ⟨b, hb_mem, mulVec_eq_zero_of_ratio_zero b h_vb_pos h_ratio_zero⟩

omit [Nonempty n] in
lemma le_any_ratio [DecidableEq n] (A : Matrix n n ℝ)
    {x : n → ℝ} (hx_nonneg : ∀ i, 0 ≤ x i) (hx_ne_zero : x ≠ 0)
    (i : n) (hi_pos : 0 < x i) :
    collatzWielandtFn A x ≤ (A *ᵥ x) i / x i := by
  dsimp [collatzWielandtFn]
  have h_supp_nonempty : ({k | 0 < x k}.toFinset).Nonempty := by
    rw [Set.toFinset_nonempty_iff, Set.nonempty_def]
    obtain ⟨j, hj_ne_zero⟩ := Function.exists_ne_zero_of_ne_zero hx_ne_zero
    exact ⟨j, lt_of_le_of_ne (hx_nonneg j) hj_ne_zero.symm⟩
  rw [dif_pos h_supp_nonempty]
  apply Finset.inf'_le
  rw [Set.mem_toFinset]
  exact hi_pos

/-- The set of values from the Collatz-Wielandt function is bounded above by the maximum row sum of A. -/
lemma bddAbove [DecidableEq n] (A : Matrix n n ℝ) (hA_nonneg : ∀ i j, 0 ≤ A i j) :
    BddAbove (collatzWielandtFn A '' {x | (∀ i, 0 ≤ x i) ∧ x ≠ 0}) := by
  use Finset.univ.sup' Finset.univ_nonempty (fun i ↦ ∑ j, A i j)
  rintro y ⟨x, ⟨hx_nonneg, hx_ne_zero⟩, rfl⟩
  obtain ⟨m, _, h_max_eq⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty x
  have h_xm_pos : 0 < x m := by
    obtain ⟨i, hi_pos⟩ : ∃ i, 0 < x i := by
      obtain ⟨j, hj⟩ := Function.exists_ne_zero_of_ne_zero hx_ne_zero
      exact ⟨j, lt_of_le_of_ne (hx_nonneg j) hj.symm⟩
    rw [← h_max_eq]
    exact lt_of_lt_of_le hi_pos (le_sup' x (Finset.mem_univ i))
  have h_le_ratio : collatzWielandtFn A x ≤ (A *ᵥ x) m / x m :=
    le_any_ratio A hx_nonneg hx_ne_zero m h_xm_pos
  have h_ratio_le : (A *ᵥ x) m / x m ≤ Finset.univ.sup' Finset.univ_nonempty (fun k ↦ ∑ l, A k l) := by
    rw [mulVec_apply, div_le_iff h_xm_pos]
    calc
      ∑ j, A m j * x j
        ≤ ∑ j, A m j * x m := by
          apply Finset.sum_le_sum; intro j _; exact mul_le_mul_of_nonneg_left (by rw [← h_max_eq]; exact le_sup' x (Finset.mem_univ j)) (hA_nonneg m j)
      _ = (∑ j, A m j) * x m := by rw [Finset.sum_mul]
      _ ≤ (Finset.univ.sup' Finset.univ_nonempty (fun k ↦ ∑ l, A k l)) * x m := by
          apply mul_le_mul_of_nonneg_right
          · exact le_sup' (fun k => ∑ l, A k l) (Finset.mem_univ m)
          · exact le_of_lt h_xm_pos
  exact le_trans h_le_ratio h_ratio_le

/-- The set of values from the Collatz-Wielandt function is non-empty. -/
lemma set_nonempty :
    (collatzWielandtFn A '' {x | (∀ i, 0 ≤ x i) ∧ x ≠ 0}).Nonempty := by
  let P_set := {x : n → ℝ | (∀ i, 0 ≤ x i) ∧ x ≠ 0}
  let x_ones : n → ℝ := fun _ ↦ 1
  have h_x_ones_in_set : x_ones ∈ P_set := by
    constructor
    · intro i; exact zero_le_one
    · intro h_zero
      have h_contra : (1 : ℝ) = 0 := by simpa [x_ones] using congr_fun h_zero (Classical.arbitrary n)
      exact one_ne_zero h_contra
  exact Set.Nonempty.image _ ⟨x_ones, h_x_ones_in_set⟩

omit [Fintype n] [Nonempty n] in
lemma smul [Fintype n] [Nonempty n] [DecidableEq n] {c : ℝ} (hc : 0 < c) (_ : ∀ i j, 0 ≤ A i j)
  {x : n → ℝ} (hx_nonneg : ∀ i, 0 ≤ x i) (hx_ne : x ≠ 0) :
  collatzWielandtFn A (c • x) = collatzWielandtFn A x := by
  dsimp [collatzWielandtFn]
  let S := {i | 0 < x i}.toFinset
  obtain ⟨i₀, hi₀⟩ := exists_pos_of_ne_zero hx_nonneg hx_ne
  have hS_nonempty : S.Nonempty := ⟨i₀, by simp [S, hi₀]⟩
  have h_supp_eq : {i | 0 < (c • x) i}.toFinset = S := by
    ext i
    simp [S, smul_eq_mul, mul_pos_iff_of_pos_left hc]
  rw [dif_pos (h_supp_eq.symm ▸ hS_nonempty), dif_pos hS_nonempty]
  refine inf'_congr (Eq.symm h_supp_eq ▸ hS_nonempty) h_supp_eq ?_
  intro i hi
  simp only [mulVec_smul, smul_eq_mul, Pi.smul_apply]
  rw [mul_div_mul_left _ _ (ne_of_gt hc)]

/-- The minimum ratio `(Ax)_i / x_i` for a positive vector `x`. -/
noncomputable def minRatio (A : Matrix n n ℝ) (x : n → ℝ) : ℝ :=
  ⨅ i, (A.mulVec x i) / x i

/-- The maximum ratio `(Ax)_i / x_i` for a positive vector `x`. -/
noncomputable def maxRatio (A : Matrix n n ℝ) (x : n → ℝ) : ℝ :=
  ⨆ i, (A.mulVec x i) / x i

/-- The Collatz-Wielandt formula for the Perron root, defined as a supremum of infima. -/
noncomputable def perronRoot (A : Matrix n n ℝ) : ℝ :=
  ⨆ (x : n → ℝ) (_ : ∀ i, 0 < x i), minRatio A x

/-- The Collatz-Wielandt formula for the Perron root, defined as an infimum of suprema. -/
noncomputable def perronRoot' (A : Matrix n n ℝ) : ℝ :=
  ⨅ (x : n → ℝ) (_ : ∀ i, 0 < x i), maxRatio A x

/-- An alternative definition of the Perron root, as the supremum of the Collatz-Wielandt function. -/
noncomputable def perronRoot_alt (A : Matrix n n ℝ) : ℝ :=
  sSup (collatzWielandtFn A '' P_set)

omit [Nonempty n] in
lemma minRatio_le_maxRatio (A : Matrix n n ℝ) (x : n → ℝ) :
    minRatio A x ≤ maxRatio A x := by
  cases isEmpty_or_nonempty n
  · simp [minRatio, maxRatio]
  · haveI : Nonempty n := inferInstance
    exact ciInf_le_ciSup (Set.finite_range _).bddBelow (Set.finite_range _).bddAbove

omit [Nonempty n] in
-- Auxiliary lemma: the sets used in sSup and sInf are nonempty
lemma min_max_sets_nonempty [Nonempty n] (A : Matrix n n ℝ) :
  ({r | ∃ x : n → ℝ, (∀ i, 0 < x i) ∧ r = minRatio A x}.Nonempty) ∧
  ({r | ∃ x : n → ℝ, (∀ i, 0 < x i) ∧ r = maxRatio A x}.Nonempty) := by
  constructor
  · use minRatio A (fun _ => 1)
    use fun _ => 1
    constructor
    · intro i; exact zero_lt_one
    · rfl
  · use maxRatio A (fun _ => 1)
    use fun _ => 1
    constructor
    · intro i; exact zero_lt_one
    · rfl

omit [Nonempty n] in
-- Auxiliary lemma: for any minimum ratio, there exists a maximum ratio that's greater
lemma forall_exists_min_le_max [Nonempty n] (A : Matrix n n ℝ) :
  ∀ r ∈ {r | ∃ x : n → ℝ, (∀ i, 0 < x i) ∧ r = minRatio A x},
    ∃ s ∈ {s | ∃ y : n → ℝ, (∀ i, 0 < y i) ∧ s = maxRatio A y}, r ≤ s := by
  intro r hr
  rcases hr with ⟨x, hx_pos, hr_eq⟩
  use maxRatio A x
  constructor
  · use x
  · rw [hr_eq]
    exact minRatio_le_maxRatio A x

theorem eq_eigenvalue_of_positive_eigenvector
  {n : Type*} [Fintype n] [DecidableEq n] [Nonempty n]
  {A : Matrix n n ℝ} {r : ℝ} {v : n → ℝ}
  (hv_pos : ∀ i, 0 < v i) (h_eig : A *ᵥ v = r • v) :
    collatzWielandtFn A v = r := by
  dsimp [collatzWielandtFn]
  have h_supp_nonempty : ({i | 0 < v i}.toFinset).Nonempty := by
    let i0 := Classical.arbitrary n
    simp
    simp_all only [filter_True, Finset.univ_nonempty]
  rw [dif_pos h_supp_nonempty]
  apply Finset.inf'_eq_of_forall_le_of_exists_le h_supp_nonempty
  · intro i hi
    let hi_pos := Set.mem_toFinset.mp hi
    have : (A *ᵥ v) i = (r • v) i := by rw [h_eig]
    rw [Pi.smul_apply, smul_eq_mul] at this
    have : (A *ᵥ v) i / v i = r := by
      rw [this]; rw [mul_div_cancel_pos_right rfl (hv_pos i)]
    rw [this]
  · use h_supp_nonempty.choose
    use h_supp_nonempty.choose_spec
    let hi_pos := Set.mem_toFinset.mp h_supp_nonempty.choose_spec
    have : (A *ᵥ v) h_supp_nonempty.choose = (r • v) h_supp_nonempty.choose := by rw [h_eig]
    rw [Pi.smul_apply, smul_eq_mul] at this
    rw [this]; rw [mul_div_cancel_pos_right rfl (hv_pos (Exists.choose h_supp_nonempty))]

lemma bddAbove_image_P_set [DecidableEq n] (A : Matrix n n ℝ) (hA_nonneg : ∀ i j, 0 ≤ A i j) :
    BddAbove (collatzWielandtFn A '' {x | (∀ i, 0 ≤ x i) ∧ x ≠ 0}) := by
  use Finset.univ.sup' Finset.univ_nonempty (fun i ↦ ∑ j, A i j)
  rintro _ ⟨x, ⟨hx_nonneg, hx_ne_zero⟩, rfl⟩
  obtain ⟨m, _, h_max_eq⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty x
  have h_xm_pos : 0 < x m := by
    obtain ⟨i, hi_pos⟩ : ∃ i, 0 < x i := by
      obtain ⟨j, hj⟩ := Function.exists_ne_zero_of_ne_zero hx_ne_zero
      exact ⟨j, lt_of_le_of_ne (hx_nonneg j) hj.symm⟩
    rw [← h_max_eq]
    exact lt_of_lt_of_le hi_pos (le_sup' x (Finset.mem_univ i))
  have h_le_ratio : collatzWielandtFn A x ≤ (A *ᵥ x) m / x m :=
    CollatzWielandt.le_any_ratio A hx_nonneg hx_ne_zero m h_xm_pos
  have h_ratio_le : (A *ᵥ x) m / x m ≤ Finset.univ.sup' Finset.univ_nonempty (fun k ↦ ∑ l, A k l) := by
    rw [mulVec_apply, div_le_iff h_xm_pos]
    calc
      ∑ j, A m j * x j
        ≤ ∑ j, A m j * x m := by
          apply Finset.sum_le_sum; intro j _; exact mul_le_mul_of_nonneg_left (by rw [← h_max_eq]; exact le_sup' x (Finset.mem_univ j)) (hA_nonneg m j)
      _ = (∑ j, A m j) * x m := by rw [Finset.sum_mul]
      _ ≤ (Finset.univ.sup' Finset.univ_nonempty (fun k ↦ ∑ l, A k l)) * x m := by
          apply mul_le_mul_of_nonneg_right
          · exact le_sup' (fun k => ∑ l, A k l) (Finset.mem_univ m)
          · exact le_of_lt h_xm_pos
  exact le_trans h_le_ratio h_ratio_le

variable {n : Type*} [Fintype n]--[Nonempty n] --[DecidableEq n]
variable {A : Matrix n n ℝ}

/-- Any eigenvalue with a strictly positive eigenvector is ≤ the Perron root. -/
theorem eigenvalue_le_perron_root_of_positive_eigenvector
    {r : ℝ} {v : n → ℝ}
    [Nonempty n] [DecidableEq n]
    (hA_nonneg : ∀ i j, 0 ≤ A i j) (_ : 0 < r)
    (hv_pos : ∀ i, 0 < v i) (h_eig : A *ᵥ v = r • v) :
    r ≤ perronRoot_alt A := by
  have hv_nonneg : ∀ i, 0 ≤ v i := fun i ↦ (hv_pos i).le
  have hv_ne_zero : v ≠ 0 := by
    intro h
    have hcontr : (0 : ℝ) < 0 := by
      have hpos := hv_pos (Classical.arbitrary n)
      simp [h] at hpos
    exact (lt_irrefl _ hcontr).elim
  have h_r : r = collatzWielandtFn A v :=
    (eq_eigenvalue_of_positive_eigenvector hv_pos h_eig).symm
  have h_le : collatzWielandtFn A v ≤ perronRoot_alt A := by
    dsimp [perronRoot_alt]
    have h_bdd : BddAbove (collatzWielandtFn A '' P_set) :=
      bddAbove_image_P_set A hA_nonneg
    apply le_csSup h_bdd
    have hv_in_P : v ∈ P_set := ⟨hv_nonneg, hv_ne_zero⟩
    exact Set.mem_image_of_mem (collatzWielandtFn A) hv_in_P
  simpa [h_r] using h_le

/-- A left eigenvector of the matrix is a right eigenvector of its transpose -/
lemma left_eigenvector_of_transpose {r : ℝ} {u : n → ℝ}
    (hu_left : u ᵥ* A = r • u) :
    Aᵀ *ᵥ u = r • u := by
  rwa [← vecMul_eq_mulVec_transpose]

/-- For any non-negative vector `w`, its Collatz–Wielandt value … -/
lemma le_eigenvalue_of_left_eigenvector [DecidableEq n]
    (hA_nonneg : ∀ i j, 0 ≤ A i j) {r : ℝ} (_ : 0 < r)
    {u : n → ℝ} (hu_pos : ∀ i, 0 < u i) (h_eig : u ᵥ* A = r • u)
    {w : n → ℝ} (hw_nonneg : ∀ i, 0 ≤ w i) (hw_ne_zero : w ≠ 0) :
    collatzWielandtFn A w ≤ r := by
  classical
  have hu_nonneg : ∀ i, 0 ≤ u i := fun i ↦ (hu_pos i).le
  -- Pointwise inequality (λr w ≤ A·w)
  have h_le_mulVec := CollatzWielandt.le_mulVec hA_nonneg hw_nonneg hw_ne_zero
  -- Multiply componentwise by u i ≥ 0 and sum to get a dot-product inequality
  have h_intermediate :
      u ⬝ᵥ ((collatzWielandtFn A w) • w) ≤ u ⬝ᵥ (A *ᵥ w) := by
    unfold dotProduct
    apply Finset.sum_le_sum
    intro i _
    have hi := h_le_mulVec i
    -- hi : ((collatzWielandtFn A w) • w) i ≤ (A *ᵥ w) i
    -- Rewrite and multiply by u i ≥ 0
    simp [Pi.smul_apply, smul_eq_mul] at hi ⊢
    exact mul_le_mul_of_nonneg_left hi (hu_nonneg i)
  -- Rewrite both sides using eigenvector relations
  have h_right :
      u ⬝ᵥ (A *ᵥ w) = r * (u ⬝ᵥ w) := by
    -- u ⬝ (A w) = (uᵀ A) ⬝ w = (r • u) ⬝ w = r * (u ⬝ w)
    have := dotProduct_mulVec u A w
    -- this : u ⬝ᵥ (A *ᵥ w) = u ᵥ* A ⬝ᵥ w
    calc
      u ⬝ᵥ (A *ᵥ w) = (u ᵥ* A) ⬝ᵥ w := this
      _ = (r • u) ⬝ᵥ w := by simp [h_eig]
      _ = r * (u ⬝ᵥ w) := by simp [smul_eq_mul]
  have h_left :
      u ⬝ᵥ ((collatzWielandtFn A w) • w)
        = (collatzWielandtFn A w) * (u ⬝ᵥ w) := by
    simp [dotProduct_smul, smul_eq_mul]
  -- Combine
  have h_dot_le :
      (collatzWielandtFn A w) * (u ⬝ᵥ w) ≤ r * (u ⬝ᵥ w) := by
    simpa [h_left, h_right] using h_intermediate
  -- Since u ⬝ w > 0 we can divide
  have h_dot_pos : 0 < u ⬝ᵥ w :=
    dotProduct_pos_of_pos_of_nonneg_ne_zero hu_pos hw_nonneg hw_ne_zero
  exact (mul_le_mul_right h_dot_pos).mp h_dot_le

/-- If v is an eigenvector of A with eigenvalue r (i.e., A *ᵥ v = r • v),
    this lemma provides the relation in the form needed for rewriting. -/
lemma mulVec_eq_smul_of_eigenvector {n : Type*} [Fintype n] [DecidableEq n]
    {A : Matrix n n ℝ} {r : ℝ} {v : n → ℝ} (h_eig : A *ᵥ v = r • v) :
    r • v = A *ᵥ v := by
  exact Eq.symm h_eig

/--
If `u` is a strictly positive left eigenvector of `A` for eigenvalue `r > 0`,
then the Perron root of `A` is less than or equal to `r`.
That is, `perronRoot_alt A ≤ r`.
-/
lemma perron_root_le_eigenvalue_of_left_eigenvector [Nonempty n] [DecidableEq n]
    (hA_nonneg : ∀ i j, 0 ≤ A i j) {r : ℝ} (hr_pos : 0 < r) {u : n → ℝ} (hu_pos : ∀ i, 0 < u i)
    (h_eig : u ᵥ* A = r • u) :
    perronRoot_alt A ≤ r := by
  dsimp [perronRoot_alt]
  apply csSup_le
  · exact CollatzWielandt.set_nonempty
  · rintro _ ⟨w, ⟨hw_nonneg, hw_ne_zero⟩, rfl⟩
    exact CollatzWielandt.le_eigenvalue_of_left_eigenvector hA_nonneg hr_pos hu_pos h_eig hw_nonneg hw_ne_zero

/--
An intermediate algebraic result for the Perron-Frobenius theorem.
If `v` is a right eigenvector of `A` for eigenvalue `r`, then for any vector `w`,
the dot product `v ⬝ᵥ (A *ᵥ w)` is equal to `r * (v ⬝ᵥ w)`.
-/
lemma dotProduct_mulVec_eq_eigenvalue_mul_dotProduct
    {r : ℝ} {v w : n → ℝ} (h_eig : Aᵀ *ᵥ v = r • v) :
    v ⬝ᵥ (A *ᵥ w) = r * (v ⬝ᵥ w) := by
  have : v ⬝ᵥ (A *ᵥ w) = v ᵥ* A ⬝ᵥ w := by exact dotProduct_mulVec v A w
  rw [this]
  have : v ᵥ* A = Aᵀ *ᵥ v := by exact vecMul_eq_mulVec_transpose A v
  rw [this]
  rw [h_eig]
  exact dotProduct_smul_left r v w

/--
If `v` is a strictly positive right eigenvector of `A` with eigenvalue `r`, then the vector
of all ones is a right eigenvector of the similarity-transformed matrix `B = D⁻¹AD`
(where `D` is `diagonal v`) with the same eigenvalue `r`.
-/
lemma ones_eigenvector_of_similarity_transform [DecidableEq n]
    {A : Matrix n n ℝ} {r : ℝ} {v : n → ℝ}
    (hv_pos : ∀ i, 0 < v i) (h_eig : A *ᵥ v = r • v) :
    (diagonal (v⁻¹) * A * diagonal v) *ᵥ (fun _ => 1) = fun _ => r := by
  let D := diagonal v
  let D_inv := diagonal (v⁻¹)
  let B := D_inv * A * D
  let ones := fun _ : n => (1 : ℝ)
  calc
    B *ᵥ ones
      = D_inv *ᵥ (A *ᵥ (D *ᵥ ones)) := by
          unfold B
          rw [← mulVec_mulVec, ← mulVec_mulVec]
    _ = D_inv *ᵥ (A *ᵥ v) := by
        have h_D_ones : D *ᵥ ones = v := by
          unfold D ones
          exact diagonal_mulVec_ones v
        rw [h_D_ones]
    _ = D_inv *ᵥ (r • v) := by rw [h_eig]
    _ = r • (D_inv *ᵥ v) := by rw [mulVec_smul]
    _ = r • ones := by
        have h_D_inv_v : D_inv *ᵥ v = ones := by
          unfold D_inv ones
          have hv_ne_zero : ∀ i, v i ≠ 0 := fun i => (hv_pos i).ne'
          exact diagonal_inv_mulVec_self hv_ne_zero
        rw [h_D_inv_v]
    _ = fun _ => r := by
        ext x
        simp only [Pi.smul_apply, smul_eq_mul, mul_one, ones]

/--
If `v` is a strictly positive right eigenvector of `A` with eigenvalue `r`, then the
similarity-transformed matrix `B = D⁻¹AD` (where `D` is `diagonal v`) has row sums equal to `r`.
-/
lemma row_sum_of_similarity_transformed_matrix [DecidableEq n] [Nonempty n]
    {A : Matrix n n ℝ} {r : ℝ} {v : n → ℝ}
    (hv_pos : ∀ i, 0 < v i) (h_eig : A *ᵥ v = r • v) :
    ∀ i, ∑ j, (Matrix.diagonal (v⁻¹) * A * Matrix.diagonal v) i j = r := by
  intro i
  let B := Matrix.diagonal (v⁻¹) * A * Matrix.diagonal v
  have row_sum_eq : ∑ j, B i j = (B *ᵥ (fun _ => 1)) i := by
    simp only [mulVec_apply, mul_one]
  rw [row_sum_eq]
  have h_B_eig := ones_eigenvector_of_similarity_transform hv_pos h_eig
  rw [h_B_eig]

/--
If a non-negative vector `x` satisfies `c • x ≤ B *ᵥ x` for a non-negative matrix `B`
whose row sums are all equal to `r`, then `c ≤ r`.
-/
lemma le_of_max_le_row_sum [Nonempty n] [DecidableEq n]
    {B : Matrix n n ℝ} {x : n → ℝ} {c r : ℝ}
    (hB_nonneg : ∀ i j, 0 ≤ B i j) (h_B_row_sum : ∀ i, ∑ j, B i j = r)
    (hx_nonneg : ∀ i, 0 ≤ x i) (hx_ne_zero : x ≠ 0) (h_le_Bx : c • x ≤ B *ᵥ x) :
    c ≤ r := by
  obtain ⟨k, -, h_k_max⟩ := Finset.exists_mem_eq_sup' (Finset.univ_nonempty) x
  have h_xk_pos : 0 < x k := by
    have h_exists_pos : ∃ i, 0 < x i := exists_pos_of_ne_zero hx_nonneg hx_ne_zero
    obtain ⟨j, hj_pos⟩ := h_exists_pos
    refine' lt_of_lt_of_le hj_pos _
    rw [← h_k_max]
    exact Finset.le_sup' (f := x) (Finset.mem_univ j)
  have h_le_k := h_le_Bx k
  simp only [Pi.smul_apply, smul_eq_mul] at h_le_k
  have h_Bx_le : (B *ᵥ x) k ≤ r * x k := by
    calc (B *ᵥ x) k
        = ∑ j, B k j * x j := by simp [mulVec_apply]
      _ ≤ ∑ j, B k j * x k := by
          apply Finset.sum_le_sum; intro j hj
          exact mul_le_mul_of_nonneg_left (by { rw [← h_k_max]; exact Finset.le_sup' x hj }) (hB_nonneg k j)
      _ = (∑ j, B k j) * x k := by rw [Finset.sum_mul]
      _ = r * x k := by rw [h_B_row_sum]
  exact (mul_le_mul_right h_xk_pos).mp (le_trans h_le_k h_Bx_le)

/--
For any non-negative vector `w`, its Collatz–Wielandt value is bounded above by a
positive eigenvalue `r` that has a strictly positive *right* eigenvector `v`.
-/
theorem le_eigenvalue_of_right_eigenvector [Nonempty n]  [DecidableEq n]
    (hA_nonneg : ∀ i j, 0 ≤ A i j) {r : ℝ} (_ : 0 < r)
    {v : n → ℝ} (hv_pos : ∀ i, 0 < v i) (h_eig : A *ᵥ v = r • v)
    {w : n → ℝ} (hw_nonneg : ∀ i, 0 ≤ w i) (hw_ne_zero : w ≠ 0) :
    collatzWielandtFn A w ≤ r := by
  let D := Matrix.diagonal v
  let D_inv := Matrix.diagonal (v⁻¹)
  let B := D_inv * A * D
  have hB_nonneg : ∀ i j, 0 ≤ B i j := by
    intro i j
    unfold B D D_inv
    simp only [mul_apply]
    apply Finset.sum_nonneg
    intro k _
    simp only [diagonal_apply]
    by_cases hik : i = k
    · by_cases hkj : k = j
      · simp [hik, hkj]
        subst hkj hik
        simp_all only [ne_eq, Finset.mem_univ, mul_nonneg_iff_of_pos_right, inv_pos, mul_nonneg_iff_of_pos_left]
      · simp [hik, hkj]
    · simp_all only [ne_eq, Finset.mem_univ, Pi.inv_apply, ite_mul, zero_mul, sum_ite_eq, ↓reduceIte, mul_ite, mul_zero]
      split
      next h =>
        subst h
        simp_all only [mul_nonneg_iff_of_pos_right, inv_pos, mul_nonneg_iff_of_pos_left]
      next h => simp_all only [le_refl]
  have h_B_row_sum := row_sum_of_similarity_transformed_matrix hv_pos h_eig
  let x := D_inv *ᵥ w
  have hx_nonneg : ∀ i, 0 ≤ x i := by
    intro i
    unfold x D_inv
    rw [mulVec_diagonal]
    exact mul_nonneg (inv_nonneg.mpr (hv_pos i).le) (hw_nonneg i)
  have hx_ne_zero : x ≠ 0 := by
    contrapose! hw_ne_zero
    have h_w_eq_Dx : w = D *ᵥ x := by
      unfold x D D_inv
      ext i
      simp only [mulVec_diagonal, mulVec_diagonal]
      have hv_ne_zero : v i ≠ 0 := (hv_pos i).ne'
      simp_all only [mul_diagonal, diagonal_mul, Pi.inv_apply, mul_nonneg_iff_of_pos_right, inv_pos,
        mul_nonneg_iff_of_pos_left, implies_true, ne_eq, isUnit_iff_ne_zero, not_false_eq_true,
        IsUnit.mul_inv_cancel_left, B, D_inv, D, x]
    rw [h_w_eq_Dx, hw_ne_zero, mulVec_zero]
  have h_le_Bx : (collatzWielandtFn A w) • x ≤ B *ᵥ x := by
    have h_le_mulVec := CollatzWielandt.le_mulVec hA_nonneg hw_nonneg hw_ne_zero
    have h_w_eq_Dx : w = D *ᵥ x := by
      unfold x D D_inv
      ext i
      simp only [mulVec_diagonal, mulVec_diagonal]
      have hv_ne_zero : v i ≠ 0 := (hv_pos i).ne'
      simp_all only [ne_eq, mul_diagonal, diagonal_mul, Pi.inv_apply, mul_nonneg_iff_of_pos_right, inv_pos,
        mul_nonneg_iff_of_pos_left, implies_true, isUnit_iff_ne_zero, not_false_eq_true, IsUnit.mul_inv_cancel_left,
        B, D_inv, D, x]
    have h_smul_le : (collatzWielandtFn A w) • w ≤ A *ᵥ w := h_le_mulVec
    have h1 : (collatzWielandtFn A w) • x = D_inv *ᵥ ((collatzWielandtFn A w) • w) := by
      rw [← mulVec_smul, h_w_eq_Dx]
    have h2 : D_inv *ᵥ (A *ᵥ w) = D_inv *ᵥ (A *ᵥ (D *ᵥ x)) := by
      rw [h_w_eq_Dx]
    have h3 : D_inv *ᵥ (A *ᵥ (D *ᵥ x)) = (D_inv * A * D) *ᵥ x := by
      rw [← mulVec_mulVec, ← mulVec_mulVec]
    rw [h1]
    have h_Dinv_nonneg : ∀ i j, 0 ≤ D_inv i j := by
      intro i j
      unfold D_inv
      rw [diagonal_apply]
      by_cases hij : i = j
      · simp only [hij, ↓reduceIte, Pi.inv_apply, inv_nonneg]
        exact le_of_lt (hv_pos j)
      · simp only [hij, ↓reduceIte, le_refl]
    intro i
    have h_comp_le : ((collatzWielandtFn A w) • w) i ≤ (A *ᵥ w) i := h_smul_le i
    have h_mulVec_mono : (D_inv *ᵥ ((collatzWielandtFn A w) • w)) i ≤ (D_inv *ᵥ (A *ᵥ w)) i := by
      simp only [mulVec_apply]
      apply Finset.sum_le_sum
      intro j _
      exact mul_le_mul_of_nonneg_left (h_le_mulVec j) (h_Dinv_nonneg i j)
    calc (D_inv *ᵥ (collatzWielandtFn A w • w)) i
      ≤ (D_inv *ᵥ (A *ᵥ w)) i := h_mulVec_mono
      _ = (D_inv *ᵥ (A *ᵥ (D *ᵥ x))) i := by rw [h_w_eq_Dx]
      _ = ((D_inv * A * D) *ᵥ x) i := by rw [← mulVec_mulVec, ← mulVec_mulVec]
      _ = (B *ᵥ x) i := rfl
  exact le_of_max_le_row_sum hB_nonneg h_B_row_sum hx_nonneg hx_ne_zero h_le_Bx

/- Any positive eigenvalue `r` with a strictly positive right eigenvector `v` is an
upper bound for the range of the Collatz-Wielandt function.
-/
theorem eigenvalue_is_ub_of_positive_eigenvector [Nonempty n]  [DecidableEq n]
    (hA_nonneg : ∀ i j, 0 ≤ A i j) {r : ℝ} (hr_pos : 0 < r)
    {v : n → ℝ} (hv_pos : ∀ i, 0 < v i) (h_eig : A *ᵥ v = r • v) :
    perronRoot_alt A ≤ r := by
  dsimp [perronRoot_alt]
  apply csSup_le (CollatzWielandt.set_nonempty (A := A))
  rintro _ ⟨w, ⟨hw_nonneg, hw_ne_zero⟩, rfl⟩
  exact CollatzWielandt.le_eigenvalue_of_right_eigenvector
    hA_nonneg hr_pos hv_pos h_eig hw_nonneg hw_ne_zero

theorem eq_perron_root_of_positive_eigenvector
    [Nonempty n] [DecidableEq n]
    {A : Matrix n n ℝ} {r : ℝ} {v : n → ℝ}
    (hA_nonneg : ∀ i j, 0 ≤ A i j)
    (hv_pos    : ∀ i, 0 < v i)
    (hr_pos    : 0 < r)
    (h_eig     : A *ᵥ v = r • v) :
    r = CollatzWielandt.perronRoot_alt (A := A) := by
  -- 1.  `r ≤ perronRoot_alt A`.
  have h₁ : r ≤ CollatzWielandt.perronRoot_alt (A := A) :=
    CollatzWielandt.eigenvalue_le_perron_root_of_positive_eigenvector
      (A := A) hA_nonneg hr_pos hv_pos h_eig
  -- 2.  `perronRoot_alt A ≤ r`.
  have h₂ : CollatzWielandt.perronRoot_alt (A := A) ≤ r :=
    CollatzWielandt.eigenvalue_is_ub_of_positive_eigenvector
      hA_nonneg hr_pos hv_pos h_eig
  exact le_antisymm h₁ h₂

lemma perronRoot'_le_maxRatio_of_min_ge_perronRoot'
    [Nonempty n] {A : Matrix n n ℝ} {x : n → ℝ}
    (hr : perronRoot' A ≤ minRatio A x) :
    perronRoot' A ≤ maxRatio A x :=
  hr.trans (minRatio_le_maxRatio A x)

/--
For a function `f` on a non-empty finite type `ι`, the indexed infimum `⨅ i, f i` is equal
to the infimum over the universal finset.
-/
lemma ciInf_eq_finset_inf' {α ι : Type*} [Fintype ι] [Nonempty ι] [ConditionallyCompleteLinearOrder α]
  (f : ι → α) :
  ⨅ i, f i = Finset.univ.inf' Finset.univ_nonempty f := by
  -- This is the symmetric version of `Finset.inf'_univ_eq_ciInf`.
  exact (Finset.inf'_univ_eq_ciInf f).symm

@[simp]
theorem Finset.sum_def {α M : Type*} [AddCommMonoid M] {s : Finset α} (f : α → M) :
  (∑ i ∈ s, f i) = s.sum f :=
rfl

/--  A finite sum of non-negative terms is strictly positive as soon as one
     summand is strictly positive.  -/
lemma Finset.sum_pos_of_nonneg_of_exists_pos {α β : Type*}
  [AddCommMonoid β] [PartialOrder β] [IsOrderedCancelAddMonoid β]
 {s : Finset α} {f : α → β}
    (h_nonneg : ∀ i ∈ s, 0 ≤ f i)
    (h_exists : ∃ i ∈ s, 0 < f i) :
    0 < ∑ i ∈ s, f i :=
  Finset.sum_pos' h_nonneg h_exists

omit [Fintype n] in
lemma maximizer_satisfies_le_mulVec
    [Fintype n] [Nonempty n] [DecidableEq n]
    (A : Matrix n n ℝ) (hA_nonneg : ∀ i j, 0 ≤ A i j) :
    let r := perronRoot_alt A
    ∃ v ∈ stdSimplex ℝ n, r • v ≤ A *ᵥ v := by
  let r := perronRoot_alt A
  obtain ⟨v, v_in_simplex, v_is_max⟩ := exists_maximizer (A := A)
  have v_ne_zero : v ≠ 0 := by
    intro hv
    have h_sum_one : (∑ i, v i) = 1 := v_in_simplex.2
    rw [hv] at h_sum_one
    simp only [Pi.zero_apply, Finset.sum_const_zero] at h_sum_one
    norm_num at h_sum_one
  have v_nonneg : ∀ i, 0 ≤ v i := v_in_simplex.1
  have r_eq : (perronRoot_alt A) = collatzWielandtFn A v := by
    dsimp [perronRoot_alt]
    apply le_antisymm
    · -- `perronRoot_alt A ≤ collatzWielandtFn A v`
      apply csSup_le set_nonempty
      rintro _ ⟨x, ⟨hx_nonneg, hx_ne_zero⟩, rfl⟩
      set s : ℝ := ∑ i, x i with hs
      have s_pos : 0 < s := by
        obtain ⟨i, hi⟩ := exists_pos_of_ne_zero hx_nonneg hx_ne_zero
        have : 0 < ∑ i, x i :=
          Finset.sum_pos_of_nonneg_of_exists_pos
            (λ j _ ↦ hx_nonneg j)
            ⟨i, Finset.mem_univ _, hi⟩
        simpa [hs] using this
      set x' : n → ℝ := s⁻¹ • x with hx'
      have hx'_in_simplex : x' ∈ stdSimplex ℝ n := by
        -- Positivity
        constructor
        · intro i
          have : 0 ≤ s⁻¹ := inv_nonneg.2 s_pos.le
          have : 0 ≤ s⁻¹ * x i := mul_nonneg this (hx_nonneg i)
          simpa [hx'] using this
        -- Sum = 1
        · have : (∑ i, x' i) = 1 := by
            simp only [hx', Pi.smul_apply, smul_eq_mul, ← Finset.mul_sum, ← hs]
            field_simp [ne_of_gt s_pos]
          exact this
      -- Maximality of `v`
      have h_max : collatzWielandtFn A x' ≤ collatzWielandtFn A v :=
        v_is_max hx'_in_simplex
      -- Scale invariance
      have h_scale : collatzWielandtFn A x = collatzWielandtFn A x' := by
        have h_smul := smul (inv_pos.mpr s_pos) hA_nonneg hx_nonneg hx_ne_zero
        rw [← hx'] at h_smul
        exact h_smul.symm
      rwa [h_scale]
    · -- `collatzWielandtFn A v ≤ perronRoot_alt A`
      apply le_csSup (bddAbove_image_P_set A hA_nonneg)
      exact Set.mem_image_of_mem _ ⟨v_nonneg, v_ne_zero⟩
  have h_le : (perronRoot_alt A) • v ≤ A *ᵥ v := by
    have : (collatzWielandtFn A v) • v ≤ A *ᵥ v :=
      le_mulVec hA_nonneg v_nonneg v_ne_zero
    simpa [r_eq] using this
  refine ⟨v, v_in_simplex, ?_⟩
  simpa [r] using h_le

/--
The conditional supremum of a non-empty, bounded above set of non-negative numbers is non-negative.
-/
lemma csSup_nonneg {s : Set ℝ} (hs_nonempty : s.Nonempty) (hs_bdd : BddAbove s)
    (hs_nonneg : ∀ x ∈ s, 0 ≤ x) :
    0 ≤ sSup s := by
  obtain ⟨y, hy_mem⟩ := hs_nonempty
  have h_y_nonneg : 0 ≤ y := hs_nonneg y hy_mem
  have h_y_le_sSup : y ≤ sSup s := le_csSup hs_bdd hy_mem
  exact le_trans h_y_nonneg h_y_le_sSup

/-- The Perron root of a non-negative matrix is non-negative. -/
lemma perronRoot_nonneg {n : Type*} [Fintype n] [Nonempty n] [DecidableEq n]
    {A : Matrix n n ℝ} (hA_nonneg : ∀ i j, 0 ≤ A i j) :
    0 ≤ perronRoot_alt A := by
  unfold perronRoot_alt
  apply csSup_nonneg
  · exact CollatzWielandt.set_nonempty
  · exact CollatzWielandt.bddAbove A hA_nonneg
  · rintro _ ⟨x, ⟨hx_nonneg, hx_ne_zero⟩, rfl⟩
    dsimp [collatzWielandtFn]
    split_ifs with h_supp_nonempty
    · apply Finset.le_inf'
      intro i hi_mem
      apply div_nonneg
      · exact mulVec_nonneg hA_nonneg hx_nonneg i
      · exact (Set.mem_toFinset.mp hi_mem).le
    · exact le_rfl

/--
If an inequality lambda•w ≤ A•w holds for a non-negative non-zero vector w,
then lambda is bounded by the Collatz-Wielandt function value for w.
This is the property that the Collatz-Wielandt function gives
the maximum lambda satisfying such an inequality.
-/
theorem le_of_subinvariant [DecidableEq n]
    {A : Matrix n n ℝ} (_ : ∀ i j, 0 ≤ A i j)
    {w : n → ℝ} (hw_nonneg : ∀ i, 0 ≤ w i) (hw_ne_zero : w ≠ 0)
    {lambda : ℝ} (h_sub : lambda • w ≤ A *ᵥ w) :
    lambda ≤ collatzWielandtFn A w := by
  obtain ⟨i, hi⟩ := exists_pos_of_ne_zero hw_nonneg hw_ne_zero
  let S := {j | 0 < w j}.toFinset
  have hS_nonempty : S.Nonempty := ⟨i, by simp [S]; exact hi⟩
  rw [collatzWielandtFn, dif_pos hS_nonempty]
  apply Finset.le_inf'
  intro j hj
  have h_j : lambda * w j ≤ (A *ᵥ w) j := by
    simp_all only [ne_eq, Set.toFinset_setOf, Finset.mem_filter, Finset.mem_univ, true_and, S]
    apply h_sub
  have hw_j_pos : 0 < w j := by simpa [S] using hj
  exact (le_div_iff₀ hw_j_pos).mpr (h_sub j)
