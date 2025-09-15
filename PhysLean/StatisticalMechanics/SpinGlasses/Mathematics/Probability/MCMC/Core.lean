import PhysLean.StatisticalMechanics.SpinGlasses.Mathematics.LinearAlgebra.Matrix.PerronFrobenius.Stochastic
namespace MCMC.Finite

open Matrix Finset

variable {n : Type*} [Fintype n]

/--
  A matrix P is (row) stochastic if it is non-negative and its rows sum to 1.
  This is the standard definition for a Markov chain transition matrix.
-/
def IsStochastic (P : Matrix n n ℝ) : Prop :=
  (∀ i j, 0 ≤ P i j) ∧ (∀ i, ∑ j, P i j = 1)

/--
  A probability distribution π (represented as a column vector in the standard simplex)
  is stationary for P if Pᵀ *ᵥ π = π.
  (This corresponds to the row vector definition π_row P = π_row).
-/
def IsStationary (P : Matrix n n ℝ) (π : stdSimplex ℝ n) : Prop :=
  Pᵀ *ᵥ π.val = π.val

/-! ### API Design (Pillar 2) -/

/-- The central object for a verified MCMC algorithm on finite spaces. (Pillar 2.1) -/
class IsMCMC [DecidableEq n] (P : Matrix n n ℝ) (π : stdSimplex ℝ n) where
  stochastic : IsStochastic P
  stationary : IsStationary P π
  irreducible : Matrix.Irreducible P
  primitive : IsPrimitive P

/-! ### Integration of the PF Theorem (Pillar 2.3 and Pillar 4, Phase 1) -/

variable [Nonempty n]

/--
  Leveraging the PF theorem to show existence and uniqueness of a stationary distribution
  for irreducible stochastic matrices.
-/
theorem exists_unique_stationary_distribution_of_irreducible
    [DecidableEq n]
    {P : Matrix n n ℝ} (h_stoch : IsStochastic P) (h_irred : Matrix.Irreducible P) :
    ∃! (π : stdSimplex ℝ n), IsStationary P π := by
  -- 1. Pᵀ is Irreducible. (Irreducibility is preserved under transposition for non-negative matrices).
  have hPT_irred : Matrix.Irreducible Pᵀ := h_irred.transpose h_stoch.1
  -- 2. Pᵀ is Column-Stochastic. (Since P is row-stochastic).
  have hPT_col_stoch : ∀ j, ∑ i, Pᵀ i j = 1 := by
    intro j
    simp [transpose_apply]
    exact h_stoch.2 j
  -- 3. Apply the PF theorem to Pᵀ.
  have h_exists := Matrix.exists_positive_eigenvector_of_irreducible_stochastic hPT_irred hPT_col_stoch
  -- 4. The result (Pᵀ *ᵥ v = v) is exactly the definition of IsStationary P π.
  simp [IsStationary]
  exact h_exists


/-- The unique stationary distribution of an irreducible stochastic matrix. -/
noncomputable def stationaryDistribution [DecidableEq n] (P : Matrix n n ℝ) (h_irred : Matrix.Irreducible P)
  (h_stoch : IsStochastic P) : stdSimplex ℝ n :=
  (Classical.choose (exists_unique_stationary_distribution_of_irreducible h_stoch h_irred).exists)

lemma stationaryDistribution_is_stationary [DecidableEq n] (P : Matrix n n ℝ) (h_irred : Matrix.Irreducible P)
  (h_stoch : IsStochastic P) :
  IsStationary P (stationaryDistribution P h_irred h_stoch) :=
  (Classical.choose_spec (exists_unique_stationary_distribution_of_irreducible h_stoch h_irred).exists)

/--
  The main theorem for Phase 1 (Pillar 2.3). If a transition matrix is stochastic,
  irreducible, and primitive, it defines a valid MCMC setup targeting its unique
  stationary distribution (which is guaranteed to exist by the PF theorem).
-/
theorem isMCMC_of_properties
    (P : Matrix n n ℝ) [DecidableEq n]
    (h_stoch : IsStochastic P)
    (h_irred : Matrix.Irreducible P)
    (h_prim : IsPrimitive P) :
    IsMCMC P (stationaryDistribution P h_irred h_stoch) :=
{
  stochastic := h_stoch,
  stationary := stationaryDistribution_is_stationary P h_irred h_stoch,
  irreducible := h_irred,
  primitive := h_prim
}

end MCMC.Finite
