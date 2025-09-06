import PhysLean.StatisticalMechanics.SpinGlasses.HopfieldNetwork.Convergence
import PhysLean.StatisticalMechanics.SpinGlasses.HopfieldNetwork.Core

/-!
Builder: derive `IsStrictlyHamiltonian (HopfieldNetwork R U)` from the *core* flip lemmas
(without duplicating the earlier hand‑written instance in `CoreBridge'`).

We use this instead of the ad‑hoc instance in `CoreBridge'` so that:
  * energy = `State.E`
  * auxPotential = remaining minus spins
  * proofs reuse `energy_diff_leq_zero` and
    `energy_lt_zero_or_pluses_increase`.
Then `CoreBridge'` can just `open` this file (or import) and not redefine the instance.
-/

open NeuralNetwork State Finset

universe uR uU
variable {R : Type uR} {U : Type uU}
variable [Field R] [LinearOrder R] [IsStrictOrderedRing R]
variable [Fintype U] [Nonempty U] [DecidableEq U]

set_option linter.unusedVariables false in
/-- Auxiliary potential = remaining minus spins (|U| - plus count). -/
@[simp] def hopfieldAux (p : Params (HopfieldNetwork R U))
    (s : (HopfieldNetwork R U).State) : ℕ :=
  Fintype.card U - s.pluses

private lemma plus_indicator_le_one
    (s : (HopfieldNetwork R U).State) (u : U) :
    (if s.act u = (1:R) then 1 else 0) ≤ (1:ℕ) := by
  by_cases h : s.act u = (1:R) <;> simp [h]

/-- Sum of the 0/1 indicators of plus spins equals the cardinality of the finite
set of sites whose activation is +1 (explicit computable formulation). -/
private lemma sum_plus_indicators_eq_filter_card
    (s : (HopfieldNetwork R U).State) :
    (∑ u : U, (if s.act u = (1:R) then 1 else 0))
      = (Finset.univ.filter (fun u : U => s.act u = (1:R))).card := by
  suffices
    H : ∀ (S : Finset U),
        (∑ u ∈ S, (if s.act u = (1:R) then 1 else 0))
          = (S.filter (fun u => s.act u = (1:R))).card by
    simp
  intro S
  refine S.induction_on ?h0 ?hstep
  · dsimp; simp_rw [@Finset.filter_empty]; rfl
  · intro a S ha hIH
    by_cases hA : s.act a = (1:R)
    · rw [@Finset.card_filter]
    · rw [@Finset.card_filter]

/-- Sum of the 0/1 indicators of plus spins is ≤ number of sites. -/
private lemma sum_plus_indicators_le (s : (HopfieldNetwork R U).State) :
    (∑ u : U, (if s.act u = (1:R) then 1 else 0))
      ≤ Fintype.card U := by
  have hEq := sum_plus_indicators_eq_filter_card (s:=s)
  have hSub :
      (Finset.univ.filter (fun u : U => s.act u = (1:R)))
        ⊆ (Finset.univ : Finset U) := by
    intro u _; exact Finset.mem_univ u
  rw [hEq]
  exact card_le_univ {u | s.act u = 1}

/-- Bound: number of plus spins ≤ number of sites. -/
private lemma pluses_le_card (s : (HopfieldNetwork R U).State) :
    s.pluses ≤ Fintype.card U := by
  unfold State.pluses
  exact sum_plus_indicators_le (s:=s)

/-- Energy Lyapunov + strict auxiliary decrease on ties (builder lemma). -/
private lemma hopfield_aux_strict
    (p : Params (HopfieldNetwork R U))
    (s : (HopfieldNetwork R U).State) (u : U)
    (hchg : s.Up p u ≠ s)
    (htie : (s.Up p u).E p = s.E p) :
    hopfieldAux p (s.Up p u) < hopfieldAux p s := by
  have hchg_act : (s.Up p u).act u ≠ s.act u := by
    intro hEq; apply hchg; exact (up_act_eq_iff_eq (wθ:=p) (s:=s) (u:=u)).mp hEq
  have hcase :=
    energy_lt_zero_or_pluses_increase (wθ:=p) (s:=s) (u:=u) hchg_act
  have hPlus : s.pluses < (s.Up p u).pluses := by
    rcases hcase with hlt | ⟨hEqE, hInc⟩
    · have : False := by
        have hlt' := hlt
        rw [htie] at hlt'
        exact lt_irrefl _ hlt'
      exact this.elim
    · exact hInc
  have hUpLe : (s.Up p u).pluses ≤ Fintype.card U := pluses_le_card (s:=s.Up p u)
  have hSLe : s.pluses ≤ Fintype.card U := pluses_le_card s
  have : Fintype.card U - (s.Up p u).pluses < Fintype.card U - s.pluses :=
    (tsub_lt_tsub_iff_left_of_le_of_le hUpLe hSLe).mpr hPlus
  simpa [hopfieldAux] using this

/-- Builder: strict Hamiltonian structure for Hopfield networks over an ordered field. -/
@[simp] noncomputable
instance instIsStrictlyHamiltonian_Hopfield :
  IsStrictlyHamiltonian (HopfieldNetwork R U) where
  energy := fun p s => s.E p
  auxPotential := hopfieldAux
  energy_is_lyapunov := by
    intro p s u
    by_cases hchg : (s.Up p u).act u ≠ s.act u
    · exact energy_diff_leq_zero (wθ:=p) (s:=s) (u:=u) hchg
    · have hAct : (s.Up p u).act u = s.act u := (not_ne_iff.mp hchg)
      have hEq : s.Up p u = s :=
        (up_act_eq_iff_eq (wθ:=p) (s:=s) (u:=u)).mp hAct
      simp only [hEq]
      exact Preorder.le_refl (E p s)
  aux_strictly_decreases_on_tie := by
    intro p s u hchg htie
    exact hopfield_aux_strict p s u hchg htie

/-! Builder: strict Hamiltonian structure for Hopfield networks over a rational field. -/
@[simp] noncomputable
instance instIsStrictlyHamiltonian_Hopfield' :
  IsStrictlyHamiltonian (HopfieldNetwork ℚ U) where
  energy := fun p s => s.E p
  auxPotential := hopfieldAux
  energy_is_lyapunov := by
    intro p s u
    by_cases hchg : (s.Up p u).act u ≠ s.act u
    · exact energy_diff_leq_zero (wθ:=p) (s:=s) (u:=u) hchg
    · have hAct : (s.Up p u).act u = s.act u := (not_ne_iff.mp hchg)
      have hEq : s.Up p u = s :=
        (up_act_eq_iff_eq (wθ:=p) (s:=s) (u:=u)).mp hAct
      simp only [hEq]
      exact Preorder.le_refl (E p s)
  aux_strictly_decreases_on_tie := by
    intro p s u hchg htie
    exact hopfield_aux_strict p s u hchg htie

/-! ### Canonical ensemble and constructive convergence -/

/-- Real-specialized strictly Hamiltonian instance (no `IsStrictOrderedRing ℝ` needed). -/
@[simp] noncomputable
instance instIsStrictlyHamiltonian_Hopfield_real
  [Fintype U] [Nonempty U] :
  IsStrictlyHamiltonian (HopfieldNetwork ℝ U) where
  energy := fun p s => s.E p
  auxPotential := hopfieldAux
  energy_is_lyapunov := by
    intro p s u
    by_cases hchg : (s.Up p u).act u ≠ s.act u
    · exact energy_diff_leq_zero (wθ:=p) (s:=s) (u:=u) hchg
    · have hAct : (s.Up p u).act u = s.act u := not_ne_iff.mp hchg
      have hEq : s.Up p u = s :=
        (up_act_eq_iff_eq (wθ:=p) (s:=s) (u:=u)).mp hAct
      simp [hEq]
  aux_strictly_decreases_on_tie := by
    intro p s u hchg htie
    exact hopfield_aux_strict p s u hchg htie

noncomputable section CE_universe0
  -- Restrict to universe-0 `U` to match `toCanonicalEnsemble.IsHamiltonian` signatures.
  variable {U : Type} [DecidableEq U] [Fintype U] [Nonempty U]

  @[simp] noncomputable
  instance instIsHamiltonian_Hopfield_real'
      : IsHamiltonian (HopfieldNetwork ℝ U) where
    energy := fun p s => s.E p
    energy_measurable := by
      intro p
      simp
    energy_is_lyapunov := by
      intro p s u
      by_cases hchg : (s.Up p u).act u ≠ s.act u
      · -- Known core Lyapunov step
        exact energy_diff_leq_zero (wθ:=p) (s:=s) (u:=u) hchg
      · have hAct : (s.Up p u).act u = s.act u := not_ne_iff.mp hchg
        have hEq  : s.Up p u = s :=
          (up_act_eq_iff_eq (wθ:=p) (s:=s) (u:=u)).mp hAct
        simp [hEq]

  /-- Canonical ensemble for parameters `p`. -/
  def CE (p : Params (HopfieldNetwork ℝ U)) :
      CanonicalEnsemble (HopfieldNetwork ℝ U).State :=
    hopfieldCE (U:=U) (σ:=ℝ) (NN:=HopfieldNetwork ℝ U) p
end CE_universe0

/-- Constructively chosen stable state reached by a fair asynchronous schedule. -/
noncomputable def stabilize (p : Params (HopfieldNetwork ℝ U))
    (s₀ : (HopfieldNetwork ℝ U).State) (useq : ℕ → U) (hf : fair useq) :
    (HopfieldNetwork ℝ U).State :=
  let ex := convergence_of_hamiltonian (NN:=HopfieldNetwork ℝ U) s₀ p useq hf
  s₀.seqStates p useq (Nat.find ex)

lemma stabilize_isStable
    (p : Params (HopfieldNetwork ℝ U))
    (s₀ : (HopfieldNetwork ℝ U).State) (useq : ℕ → U) (hf : fair useq) :
    (stabilize (p:=p) s₀ useq hf).isStable p := by
  simpa [stabilize] using
    (Nat.find_spec (convergence_of_hamiltonian (NN:=HopfieldNetwork ℝ U) s₀ p useq hf))

/-! ### Propagation explanation

Importing this file yields:
* `IsHamiltonian (HN)` ⇒ access to canonical ensemble via `CE`.
* `IsStrictlyHamiltonian (HN)` ⇒ deterministic convergence theorem
  `convergence_of_hamiltonian` and helper `stabilize`.
* Detailed-balance / Gibbs results (in other files) become applicable
  once an `EnergySpec'` is supplied (or via generic two–state bridges).

Core stays computable; only this layer uses (noncomputable) real analysis
and finite type enumeration of states.
-/
