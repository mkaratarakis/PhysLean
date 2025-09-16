import Mathlib
import PhysLean.StatisticalMechanics.SpinGlasses.Mathematics.LinearAlgebra.Matrix.PerronFrobenius.Multiplicity
import PhysLean.StatisticalMechanics.SpinGlasses.Mathematics.Probability.MCMC.Core

noncomputable section
open Matrix Finset MCMC.Finite
open scoped Matrix

namespace Matrix

variable {n : Type*} [Fintype n]

/-!
Total variation on the finite simplex and the Dobrushin coefficient
for row-stochastic kernels.
-/

/-- Total variation distance on the finite simplex (sup over sets = 1/2 L1). -/
def tvDist (p q : n → ℝ) : ℝ :=
  (∑ j, |p j - q j|) / 2

lemma tvDist_nonneg (p q : n → ℝ) : 0 ≤ tvDist p q := by
  have : 0 ≤ ∑ j, |p j - q j| := by
    exact sum_nonneg (fun _ _ => abs_nonneg _)
  have h2 : 0 ≤ (2 : ℝ) := by norm_num
  simpa [tvDist, div_eq_mul_inv, mul_comm]-- using (mul_nonneg_of_nonneg_of_nonneg this (inv_nonneg.mpr h2))

/-- For finite families with equal total mass, each coordinate deviation is bounded by TV. -/
lemma coord_abs_le_tvDist_of_eq_sum [DecidableEq n] (p q : n → ℝ)
    (hsum : ∑ j, p j = ∑ j, q j) (j : n) :
    |p j - q j| ≤ tvDist p q := by
  have hx_sum0 : ∑ t, (p t - q t) = 0 := by
    simp [sum_sub_distrib, hsum]
  -- Split the L1 sum into the `j`-term and the rest.
  have hsplit :
      ∑ t, |p t - q t|
        = |p j - q j| + ∑ t ∈ (Finset.univ.erase j), |p t - q t| := by
    have : (Finset.univ : Finset n) = insert j ((Finset.univ).erase j) := by
      simp
    calc
      ∑ t, |p t - q t|
          = ∑ t ∈ insert j ((Finset.univ).erase j), |p t - q t| := by
          rw [this]
          exact
            Eq.symm
              (sum_congr (congrArg (insert j) (congrFun (congrArg erase (id (Eq.symm this))) j))
                fun x => congrFun rfl)
      _ = |p j - q j| + ∑ t ∈ ((Finset.univ).erase j), |p t - q t| := by
            simp [Finset.mem_univ]
  -- The sum of the rest terms equals the negative of the j-term (since the total sum is zero).
  have hrest_sum :
      ∑ t ∈ (Finset.univ.erase j), (p t - q t) = (q j - p j) := by
    -- Sum over `univ` is zero; isolate the `j`-term.
    have huniv_split :
        ∑ t, (p t - q t)
          = (p j - q j) + ∑ t ∈ (Finset.univ.erase j), (p t - q t) := by
      have : (Finset.univ : Finset n) = insert j ((Finset.univ).erase j) := by
        simp
      calc
        ∑ t, (p t - q t)
            = ∑ t ∈ insert j ((Finset.univ).erase j), (p t - q t) :=
              congrFun (congrArg Finset.sum this) fun t => p t - q t
        _ = (p j - q j) + ∑ t ∈ ((Finset.univ).erase j), (p t - q t) := by
              simp [Finset.mem_univ]
              grind
    have hx0' : (p j - q j) + ∑ t ∈  (Finset.univ.erase j), (p t - q t) = 0 := by
      simpa [huniv_split] using hx_sum0
    -- Rearranging gives the claim.
    have := eq_neg_of_add_eq_zero_left hx0'
    simpa [sub_eq_add_neg, add_comm] using this
  -- Triangle inequality on the "rest" yields a lower bound on the L1 sum:
  have hrest_abs_le :
      |∑ t ∈  (Finset.univ.erase j), (p t - q t)|
        ≤ ∑ t ∈  (Finset.univ.erase j), |p t - q t| :=
    (abs_sum_le_sum_abs _ _)
  -- Combine to get a lower bound on the L1 sum.
  have hL1_ge :
      ∑ t, |p t - q t| ≥ |p j - q j| + |q j - p j| := by
    calc
      ∑ t, |p t - q t|
          = |p j - q j| + ∑ t ∈  (Finset.univ.erase j), |p t - q t| := hsplit
      _ ≥ |p j - q j| + |∑ t ∈  (Finset.univ.erase j), (p t - q t)| := by
            gcongr
      _ = |p j - q j| + |q j - p j| := by simp [hrest_sum]
  -- Since |qj - pj| = |pj - qj|, the RHS is 2 * |pj - qj|.
  have h2mul : (∑ t, |p t - q t|) ≥ 2 * |p j - q j| := by
    simpa [two_mul, abs_sub_comm] using hL1_ge
  -- Divide both sides by 2 > 0 to conclude.
  have h2pos : (0 : ℝ) < 2 := by norm_num
  have h_div : |p j - q j| ≤ (∑ t, |p t - q t|) / 2 := (le_div_iff₀' h2pos).mpr h2mul
  simpa [tvDist, div_eq_mul_inv, mul_comm] using h_div

/-- The row `i` of a row-stochastic matrix seen as a probability vector. -/
def rowDist (P : Matrix n n ℝ) (i : n) : n → ℝ := fun j => P i j

/-- Dobrushin coefficient δ(P) = sup over pairs of rows of TV distance. -/
def dobrushinCoeff (P : Matrix n n ℝ) : ℝ :=
  sSup { d | ∃ i i' : n, d = tvDist (rowDist P i) (rowDist P i') }

lemma dobrushinCoeff_nonneg [Nonempty n] (P : Matrix n n ℝ) : 0 ≤ dobrushinCoeff P := by
  -- Identify the set as a finite range to get boundedness and nonemptiness.
  let f : (n × n) → ℝ := fun p => tvDist (rowDist P p.1) (rowDist P p.2)
  have hset_eq : { d | ∃ i i' : n, d = tvDist (rowDist P i) (rowDist P i') }
                  = Set.range f := by
    ext d; constructor
    · intro h; rcases h with ⟨i, i', rfl⟩; exact ⟨⟨i, i'⟩, rfl⟩
    · intro h; rcases h with ⟨⟨i, i'⟩, rfl⟩; exact ⟨i, i', rfl⟩
  have hfin : (Set.range f).Finite := (Set.finite_range f)
  have hmem : 0 ∈ Set.range f := by
    -- Take i = i', tvDist = 0
    let i0 : n := Classical.arbitrary n
    refine ⟨⟨i0, i0⟩, ?_⟩
    simp [f, rowDist, tvDist, sub_self, abs_zero, sum_const_zero]
  -- Conclude 0 ≤ sSup by `le_csSup` on a finite set.
  have hbdd : BddAbove (Set.range f) := hfin.bddAbove
  simpa [dobrushinCoeff, hset_eq] using le_csSup hbdd hmem

/-- Contraction in TV under a row-stochastic kernel (with Dobrushin's coefficient). -/
lemma tvDist_contract [Nonempty n]
    (P : Matrix n n ℝ) --(hP : MCMC.Finite.IsStochastic P)
    (p q : n → ℝ)-- (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 ≤ q j)
    (hp1 : ∑ j, p j = 1) (hq1 : ∑ j, q j = 1) :
    tvDist ((fun j => ∑ k, p k * P k j)) ((fun j => ∑ k, q k * P k j))
      ≤ dobrushinCoeff P * tvDist p q := by
  -- set r := p - q, a_j := ∑_k r_k P_{k j}
  let r : n → ℝ := fun k => p k - q k
  have hsum_r : ∑ k, r k = 0 := by
    simp [r, sum_sub_distrib, hp1, hq1]
  let a : n → ℝ := fun j => ∑ k, r k * P k j
  -- pick signs so that s j * a j = |a j|
  let s : n → ℝ := fun j => if 0 ≤ a j then 1 else -1
  have hs_abs : ∀ j, |s j| = 1 := by
    intro j; by_cases h : 0 ≤ a j
    · simp [s, h]
    · have : a j < 0 := lt_of_not_ge h
      simp [s, h]
  have hsum_abs_eq : ∑ j, |a j| = ∑ j, s j * a j := by
    apply sum_congr rfl; intro j _
    by_cases h : 0 ≤ a j
    · simp [s, h, abs_of_nonneg h]
    · have : a j < 0 := lt_of_not_ge h
      have hnn : a j ≤ 0 := le_of_lt this
      aesop
  -- swap the sums: ∑_j s j * a j = ∑_k r k * g k
  let g : n → ℝ := fun k => ∑ j, s j * P k j
  have hswap : ∑ j, s j * a j = ∑ k, r k * g k := by
    unfold a g
    calc
      ∑ j, s j * ∑ k, r k * P k j
          = ∑ j, ∑ k, s j * (r k * P k j) := by
                simp [mul_sum]
      _ = ∑ j, ∑ k, r k * (s j * P k j) := by
                apply sum_congr rfl; intro j _; simp [mul_left_comm, mul_assoc, mul_comm]
      _ = ∑ k, ∑ j, r k * (s j * P k j) := by
                simpa using
                  (Finset.sum_comm (s := (Finset.univ : Finset n))
                    (t := (Finset.univ : Finset n))
                    (f := fun j k => r k * (s j * P k j)))
      _ = ∑ k, r k * ∑ j, s j * P k j := by
                -- factor `r k` out of the inner sum
                simp [mul_sum]
  -- oscillation bound for g via rows of P
  have g_diff_le : ∀ k ℓ, |g k - g ℓ| ≤ 2 * dobrushinCoeff P := by
    intro k ℓ
    have : |g k - g ℓ| ≤ ∑ j, |P k j - P ℓ j| := by
      have : g k - g ℓ = ∑ j, s j * (P k j - P ℓ j) := by
        simp [g, sum_sub_distrib, mul_sub]
      calc
        |g k - g ℓ| = |∑ j, s j * (P k j - P ℓ j)| := by simp [this]
        _ ≤ ∑ j, |s j * (P k j - P ℓ j)| := by
              simpa using
                (abs_sum_le_sum_abs
                  (s := (Finset.univ : Finset n))
                  (f := fun j => s j * (P k j - P ℓ j)))
        _ = ∑ j, |s j| * |P k j - P ℓ j| := by
              apply sum_congr rfl; intro j _; simp [abs_mul]
        _ = ∑ j, |P k j - P ℓ j| := by
              simp [hs_abs]
    -- relate ∑ |Pk - Pℓ| to 2 * tvDist(row k, row ℓ) ≤ 2 * δ(P)
    -- prepare `le_csSup` on the finite range
    let f : (n × n) → ℝ := fun p => tvDist (rowDist P p.1) (rowDist P p.2)
    have hset_eq :
      { d | ∃ i i' : n, d = tvDist (rowDist P i) (rowDist P i') } = Set.range f := by
      ext d; constructor
      · intro h; rcases h with ⟨i, i', rfl⟩; exact ⟨⟨i, i'⟩, rfl⟩
      · intro h; rcases h with ⟨⟨i, i'⟩, rfl⟩; exact ⟨i, i', rfl⟩
    have hbdd : BddAbove (Set.range f) := (Set.finite_range f).bddAbove
    have h_tv_le : tvDist (rowDist P k) (rowDist P ℓ) ≤ dobrushinCoeff P := by
      have : tvDist (rowDist P k) (rowDist P ℓ) ∈ Set.range f := ⟨⟨k, ℓ⟩, rfl⟩
      simpa [dobrushinCoeff, hset_eq] using (le_csSup hbdd this)
    have : |g k - g ℓ| ≤ 2 * tvDist (rowDist P k) (rowDist P ℓ) := by
      simpa [two_mul, tvDist, rowDist, div_eq_mul_inv, mul_comm] using this
    exact this.trans (by
      have : tvDist (rowDist P k) (rowDist P ℓ) ≤ dobrushinCoeff P := h_tv_le
      have hnonneg : 0 ≤ 2 := by norm_num
      simp [*])
  -- choose kmax,kmin achieving max/min of g
  obtain ⟨kmax, _, hkmax⟩ :=
    Finset.exists_max_image (s := (Finset.univ : Finset n)) (f := fun k => g k)
      (by simp)
  obtain ⟨kmin, _, hkmin⟩ :=
    Finset.exists_min_image (s := (Finset.univ : Finset n)) (f := fun k => g k)
      (by simp)
  have h_le_max : ∀ k, g k ≤ g kmax := by intro k; exact hkmax k (by simp)
  have h_ge_min : ∀ k, g kmin ≤ g k := by intro k; exact hkmin k (by simp)
  -- positive/negative parts of r
  let rpos : n → ℝ := fun k => max (r k) 0
  let rneg : n → ℝ := fun k => max (-r k) 0
  have hrpos_nonneg : ∀ k, 0 ≤ rpos k := by
    intro k; have : 0 ≤ max (0:ℝ) (r k) := le_max_left _ _
    simp [rpos]
  have hrneg_nonneg : ∀ k, 0 ≤ rneg k := by
    intro k; have : 0 ≤ max (0:ℝ) (-r k) := le_max_left _ _
    simp [rneg]
  have h_r_decomp : ∀ k, r k = rpos k - rneg k := by
    intro k; by_cases hk : 0 ≤ r k
    · have : -r k ≤ 0 := neg_nonpos.mpr hk
      simp [rpos, rneg, hk, this]
    · have hk' : r k ≤ 0 := le_of_lt (lt_of_not_ge hk)
      have hneg' : 0 ≤ -r k := neg_nonneg.mpr hk'
      simp [rpos, rneg, hk', hneg', sub_eq_add_neg, add_comm]
  have hsum_pos_eq_neg : ∑ k, rpos k = ∑ k, rneg k := by
    have hsum : (∑ k, rpos k) - (∑ k, rneg k) = 0 := by
      simpa [h_r_decomp, sum_sub_distrib] using hsum_r
    exact sub_eq_zero.mp hsum
  -- express ∑ |r| as 2α
  have h_abs_split : ∀ k, |r k| = rpos k + rneg k := by
    intro k; by_cases hk : 0 ≤ r k
    · have : -r k ≤ 0 := neg_nonpos.mpr hk
      simp [rpos, rneg, hk, this, abs_of_nonneg]
    · have hk' : r k ≤ 0 := le_of_lt (lt_of_not_ge hk)
      have hneg' : 0 ≤ -r k := neg_nonneg.mpr hk'
      simp [rpos, rneg, hk', hneg', abs_of_nonpos, add_comm]
  have hsum_abs : ∑ k, |r k| = 2 * (∑ k, rpos k) := by
    calc
      ∑ k, |r k| = ∑ k, (rpos k + rneg k) := by
          apply sum_congr rfl; intro k _; simp [h_abs_split k]
      _ = (∑ k, rpos k) + (∑ k, rneg k) := by
          simp [sum_add_distrib]
      _ = (∑ k, rpos k) + (∑ k, rpos k) := by
          simp [hsum_pos_eq_neg]
      _ = 2 * (∑ k, rpos k) := by ring
  -- Bound ∑ r·g via α := ∑ rpos = ∑ rneg
  let α : ℝ := ∑ k, rpos k
  have h_sum_pos_le : ∑ k, rpos k * g k ≤ (g kmax) * α := by
    calc
      ∑ k, rpos k * g k
          ≤ ∑ k, rpos k * (g kmax) := by
            apply sum_le_sum; intro k _; exact mul_le_mul_of_nonneg_left (h_le_max k) (hrpos_nonneg k)
      _ = (g kmax) * ∑ k, rpos k := by
        simp [mul_comm]
        exact Eq.symm (sum_mul univ (fun i => max (r i) 0) (g kmax))
      _ = (g kmax) * α := rfl
  have h_sum_neg_ge : ∑ k, rneg k * g k ≥ (g kmin) * α := by
    -- rewrite (g kmin) * α using ∑ rneg via hsum_pos_eq_neg, then compare termwise
    have hα : α = ∑ k, rneg k := by
      simpa [α] using hsum_pos_eq_neg
    have h : (g kmin) * α ≤ ∑ k, rneg k * g k := by
      calc
        (g kmin) * α = ∑ k, rneg k * (g kmin) := by
          simp [mul_comm, hα, sum_mul]
        _ ≤ ∑ k, rneg k * g k := by
          apply sum_le_sum; intro k _; exact mul_le_mul_of_nonneg_left (h_ge_min k) (hrneg_nonneg k)
    simpa using h
  have h_rg_le : ∑ k, r k * g k ≤ (g kmax - g kmin) * α := by
    -- combine the previous bounds and rewrite both sides
    have h' :
        ∑ k, rpos k * g k - ∑ k, rneg k * g k
          ≤ g kmax * α - g kmin * α :=
      sub_le_sub h_sum_pos_le h_sum_neg_ge
    have h'' :
        ∑ k, rpos k * g k - ∑ k, rneg k * g k
          ≤ (g kmax - g kmin) * α := by
      have hR : g kmax * α - g kmin * α = (g kmax - g kmin) * α := by ring
      simpa [hR] using h'
    have hL :
        ∑ k, r k * g k
          = ∑ k, rpos k * g k - ∑ k, rneg k * g k := by
      calc
        ∑ k, r k * g k
            = ∑ k, (rpos k - rneg k) * g k := by
              apply sum_congr rfl; intro k _; simp [h_r_decomp k]
        _ = ∑ k, (rpos k * g k - rneg k * g k) := by
              apply sum_congr rfl; intro k _; rw [@mul_sub_right_distrib]
        _ = ∑ k, rpos k * g k - ∑ k, rneg k * g k := by
              simp [sum_sub_distrib]
    simpa [hL] using h''
  -- from ∑ |a| = ∑ s·a = ∑ r·g, obtain upper bound by oscillation
  have h_sum_bound : ∑ j, |a j| ≤ (g kmax - g kmin) / 2 * ∑ k, |r k| := by
    have hα : α = (∑ k, |r k|) / 2 := by
      have h2pos : (0 : ℝ) < 2 := by norm_num
      have : 2 * α = ∑ k, |r k| := by
        simp [α, hsum_abs, two_mul]-- using (eq_comm.mp (by simp [hsum_abs, two_mul, α]))
      -- safer derivation:
      have : ∑ k, |r k| = 2 * α := by simpa [α] using hsum_abs
      have hnonneg : 0 ≤ (2 : ℝ) := by norm_num
      -- divide both sides by 2
      calc
        α = (2 * α) / 2 := by field_simp
        _ = (∑ k, |r k|) / 2 := by simp [this]
    -- ∑ |a| = ∑ r·g
    have : ∑ j, |a j| = ∑ k, r k * g k := by
      simpa [hswap] using hsum_abs_eq
    calc
      ∑ j, |a j| = ∑ k, r k * g k := this
      _ ≤ (g kmax - g kmin) * α := h_rg_le
      _ = ((g kmax - g kmin) / 2) * (∑ k, |r k|) := by
            have : α = (∑ k, |r k|) / 2 := hα
            simp [this, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]
  -- relate oscillation of g to δ(P)
  have h_osc_le : g kmax - g kmin ≤ 2 * dobrushinCoeff P := by
    have := g_diff_le kmax kmin
    -- |g kmax - g kmin| = g kmax - g kmin since max ≥ min
    have hge : g kmin ≤ g kmax := h_ge_min kmax
    have : |g kmax - g kmin| = g kmax - g kmin := by
      simp [abs_of_nonneg (sub_nonneg.mpr hge)]
    simp; grind
  -- finish: rewrite both sides with tvDist and divide by 2
  have hLHS : tvDist (fun j => ∑ k, p k * P k j) (fun j => ∑ k, q k * P k j)
            = (∑ j, |a j|) / 2 := by
    simp only [tvDist, a, r, sub_eq_add_neg]
    congr 1
    dsimp [sum_neg_distrib, sum_add_distrib]
    congr 1
    ext j
    dsimp [sum_add_distrib, mul_neg, sum_neg_distrib]
    ring_nf
    simp
  have hR_r : tvDist p q = (∑ k, |r k|) / 2 := by
    simp [tvDist, r]
  -- use (gmax - gmin)/2 ≤ δ(P) to scale the bound
  have hcoef : (g kmax - g kmin) / 2 ≤ dobrushinCoeff P := by
    -- from h_osc_le: g kmax - g kmin ≤ 2 * dobrushinCoeff P
    have h2pos : (0 : ℝ) < 2 := by norm_num
    rwa [div_le_iff h2pos, mul_comm]
  -- Combine `h_sum_bound` with `hcoef` to bound `∑ |a|` by `δ(P) * ∑ |r|`.
  have h_mul : (∑ j, |a j|) ≤ dobrushinCoeff P * ∑ k, |r k| := by
    -- enlarge the right-hand side of `h_sum_bound` using `hcoef`
    have S_nonneg : 0 ≤ ∑ k, |r k| :=
      sum_nonneg (by
        intro _ _
        exact abs_nonneg _)
    have h_temp :
        ((g kmax - g kmin) / 2) * (∑ k, |r k|) ≤
          dobrushinCoeff P * (∑ k, |r k|) :=
      mul_le_mul_of_nonneg_right hcoef S_nonneg
    exact h_sum_bound.trans h_temp

  -- Divide both sides by `2` (equivalently multiply by `1/2`).
  have h_div : (∑ j, |a j|) / 2 ≤ (dobrushinCoeff P * ∑ k, |r k|) / 2 := by
    have : (1 / 2 : ℝ) * (∑ j, |a j|) ≤
           (1 / 2 : ℝ) * (dobrushinCoeff P * ∑ k, |r k|) :=
      mul_le_mul_of_nonneg_left h_mul (by norm_num)
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using this

  -- Rewrite both sides via `hLHS` and `hR_r` and conclude the contraction bound.
  have : tvDist (fun j => ∑ k, p k * P k j)
                (fun j => ∑ k, q k * P k j)
        ≤ dobrushinCoeff P * tvDist p q := by
    rw [hLHS, hR_r, ← mul_div_assoc]
    exact h_div
  exact this

open MCMC.Finite

/-- Submultiplicativity of the Dobrushin coefficient. -/
lemma dobrushinCoeff_mul [DecidableEq n] (P Q : Matrix n n ℝ)
    [Nonempty n]
    (hP : IsStochastic P) (_ : IsStochastic Q) :
    dobrushinCoeff (P * Q) ≤ dobrushinCoeff P * dobrushinCoeff Q := by
  classical
  -- rewrite δ(P*Q) as sSup over a finite range
  let fPQ : (n × n) → ℝ := fun p => tvDist (rowDist (P * Q) p.1) (rowDist (P * Q) p.2)
  have hset_eq_PQ :
      { d | ∃ i i' : n, d = tvDist (rowDist (P * Q) i) (rowDist (P * Q) i') }
        = Set.range fPQ := by
    ext d; constructor
    · intro h; rcases h with ⟨i, i', rfl⟩; exact ⟨⟨i, i'⟩, rfl⟩
    · intro h; rcases h with ⟨⟨i, i'⟩, rfl⟩; exact ⟨i, i', rfl⟩
  have hbddPQ : BddAbove (Set.range fPQ) := (Set.finite_range fPQ).bddAbove
  -- show every element in the range is ≤ δ(P) * δ(Q)
  have hforall :
      ∀ d ∈ Set.range fPQ, d ≤ dobrushinCoeff P * dobrushinCoeff Q := by
    intro d hd
    rcases hd with ⟨⟨i, i'⟩, rfl⟩
    -- rows of P have total mass 1 by stochasticity
    have hp1 : ∑ j, rowDist P i j = 1 := by simpa [rowDist] using hP.2 i
    have hq1 : ∑ j, rowDist P i' j = 1 := by simpa [rowDist] using hP.2 i'
    -- contract by Q
    have hcontract :
        tvDist (rowDist (P * Q) i) (rowDist (P * Q) i')
          ≤ dobrushinCoeff Q * tvDist (rowDist P i) (rowDist P i') := by
      simpa [rowDist, Matrix.mul_apply] using
        (tvDist_contract (P := Q) (p := rowDist P i) (q := rowDist P i') (hp1 := hp1) (hq1 := hq1))
    -- bound tvDist among rows of P by δ(P)
    let fP : (n × n) → ℝ := fun p => tvDist (rowDist P p.1) (rowDist P p.2)
    have hset_eq_P :
      { d | ∃ i i' : n, d = tvDist (rowDist P i) (rowDist P i') } = Set.range fP := by
      ext d; constructor
      · intro h; rcases h with ⟨i, i', rfl⟩; exact ⟨⟨i, i'⟩, rfl⟩
      · intro h; rcases h with ⟨⟨i, i'⟩, rfl⟩; exact ⟨i, i', rfl⟩
    have hbddP : BddAbove (Set.range fP) := (Set.finite_range fP).bddAbove
    have hleP : tvDist (rowDist P i) (rowDist P i') ≤ dobrushinCoeff P := by
      have hx : tvDist (rowDist P i) (rowDist P i') ∈ Set.range fP := ⟨⟨i, i'⟩, rfl⟩
      simpa [dobrushinCoeff, hset_eq_P] using le_csSup hbddP hx
    have hnonnegQ : 0 ≤ dobrushinCoeff Q := dobrushinCoeff_nonneg (P := Q)
    have := hcontract.trans (mul_le_mul_of_nonneg_left hleP hnonnegQ)
    simpa [mul_comm] using this
  -- take supremum over the finite range
  have hnonemptyPQ : (Set.range fPQ).Nonempty := by
    classical
    let i0 : n := Classical.arbitrary n
    exact ⟨fPQ ⟨i0, i0⟩, ⟨⟨i0, i0⟩, rfl⟩⟩
  have : sSup (Set.range fPQ) ≤ dobrushinCoeff P * dobrushinCoeff Q :=
    csSup_le hnonemptyPQ hforall
  simpa [dobrushinCoeff, hset_eq_PQ] using this

/-- Power bound for the Dobrushin coefficient. -/
lemma dobrushinCoeff_pow [DecidableEq n] (P : Matrix n n ℝ) [Nonempty n] (hP : MCMC.Finite.IsStochastic P) (k : ℕ) :
    dobrushinCoeff (P^k) ≤ (dobrushinCoeff P)^k := by
  classical
  induction' k with k ih
  · -- base: δ(I) ≤ 1 = (δ P)^0
    have hpair :
        ∀ i i' : n,
          tvDist (rowDist (1 : Matrix n n ℝ) i) (rowDist (1 : Matrix n n ℝ) i') ≤ 1 := by
      intro i i'
      have hpt :
          ∀ j,
            |(1 : Matrix n n ℝ) i j - (1 : Matrix n n ℝ) i' j|
              ≤ |(1 : Matrix n n ℝ) i j| + |(1 : Matrix n n ℝ) i' j| := by
        intro j
        simpa [sub_eq_add_neg] using
          (abs_add ((1 : Matrix n n ℝ) i j) (-(1 : Matrix n n ℝ) i' j))
      have hsum_le :
          ∑ j, |(1 : Matrix n n ℝ) i j - (1 : Matrix n n ℝ) i' j|
            ≤ ∑ j, (|(1 : Matrix n n ℝ) i j| + |(1 : Matrix n n ℝ) i' j|) := by
        apply sum_le_sum
        intro j _
        simpa using hpt j
      -- L1 norm of a row of the identity is 1
      have hsum_abs_one (i : n) : ∑ j, |(1 : Matrix n n ℝ) i j| = 1 := by
        classical
        -- Turn the absolute values into the same indicator form.
        have habs : ∀ j, |(1 : Matrix n n ℝ) i j| = (if i = j then (1 : ℝ) else 0) := by
          intro j
          by_cases h : i = j
          · simp [Matrix.one_apply, h]
          · simp [h]
        have hsum_eq :
            ∑ j, |(1 : Matrix n n ℝ) i j| = ∑ j, (if i = j then (1 : ℝ) else 0) := by
          apply sum_congr rfl
          intro j _
          simpa using habs j
        -- The sum of the indicator over univ is 1.
        have hsum_ind : ∑ j, (if i = j then (1 : ℝ) else 0) = 1 := by
          -- rewrite to (j = i) to use simp
          simp
        simp [hsum_eq]
      have : (∑ j, |(1 : Matrix n n ℝ) i j - (1 : Matrix n n ℝ) i' j|) / 2 ≤ 1 := by
        have h2 : (0 : ℝ) < 2 := by norm_num
        -- First bound the numerator by 2, then divide by 2 > 0
        have hnum :
            ∑ j, |(1 : Matrix n n ℝ) i j - (1 : Matrix n n ℝ) i' j| ≤ 2 := by
          have hbound :
              ∑ j, |(1 : Matrix n n ℝ) i j - (1 : Matrix n n ℝ) i' j|
                ≤ (∑ j, |(1 : Matrix n n ℝ) i j|) + (∑ j, |(1 : Matrix n n ℝ) i' j|) := by
            simpa [sum_add_distrib] using hsum_le
          have hx : (∑ j, |(1 : Matrix n n ℝ) i j|) + (∑ j, |(1 : Matrix n n ℝ) i' j|) = (2 : ℝ) := by
            -- reduce to 1 + 1 = 2
            have h12 : (1 : ℝ) + 1 = 2 := by norm_num
            simpa [hsum_abs_one i, hsum_abs_one i'] using h12
          simpa [hx] using hbound
        -- Divide the inequality by 2 > 0
        exact (div_le_iff h2).mpr (by simpa [one_mul] using hnum)
      simpa [tvDist, rowDist] using this
    let fId : (n × n) → ℝ :=
      fun p => tvDist (rowDist (1 : Matrix n n ℝ) p.1) (rowDist (1 : Matrix n n ℝ) p.2)
    have hforall : ∀ d ∈ Set.range fId, d ≤ 1 := by
      intro d hd
      rcases hd with ⟨⟨i, i'⟩, rfl⟩
      simpa using hpair i i'
    have hnonempty : (Set.range fId).Nonempty := by
      let i0 : n := Classical.arbitrary n
      exact ⟨fId ⟨i0, i0⟩, ⟨⟨i0, i0⟩, rfl⟩⟩
    have hset_eqId :
        { d | ∃ i i' : n, d = tvDist (rowDist (1 : Matrix n n ℝ) i) (rowDist (1 : Matrix n n ℝ) i') }
          = Set.range fId := by
      ext d; constructor
      · intro h; rcases h with ⟨i, i', rfl⟩; exact ⟨⟨i, i'⟩, rfl⟩
      · intro h; rcases h with ⟨⟨i, i'⟩, rfl⟩; exact ⟨i, i', rfl⟩
    have : sSup (Set.range fId) ≤ 1 := csSup_le hnonempty hforall
    simpa [dobrushinCoeff, pow_zero, rowDist, hset_eqId] using this
  · -- step: δ(P^(k+1)) ≤ δ(P^k) * δ(P) ≤ (δ P)^(k+1)
    have hPow : IsStochastic (P^k) := by
      simpa using (isStochastic_pow (P := P) (hP := hP) k)
    have hmul :
        dobrushinCoeff (P^(k+1)) ≤ dobrushinCoeff (P^k) * dobrushinCoeff P := by
      simpa [pow_succ] using
        (dobrushinCoeff_mul (P := P^k) (Q := P) (hP := hPow) hP)
    have hnonneg : 0 ≤ dobrushinCoeff P := dobrushinCoeff_nonneg (P := P)
    have := hmul.trans (mul_le_mul_of_nonneg_right ih hnonneg)
    simpa [pow_succ, mul_left_comm, mul_comm, mul_assoc] using this

/-- Entrywise deviation is bounded by TV distance when the totals match. -/
lemma entry_abs_le_tvDist_of_rows [DecidableEq n]
    (P : Matrix n n ℝ) (i : n) (x : n → ℝ) (j : n)
    (hsum : ∑ t, rowDist P i t = ∑ t, x t) :
    |(P i j) - x j| ≤ tvDist (rowDist P i) x := by
  simpa [rowDist] using
    coord_abs_le_tvDist_of_eq_sum (p := rowDist P i) (q := x) (hsum := by simpa [rowDist] using hsum) (j := j)

end Matrix
