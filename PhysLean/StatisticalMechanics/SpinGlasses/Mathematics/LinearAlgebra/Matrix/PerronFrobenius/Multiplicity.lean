import Mathlib.Order.CompletePartialOrder
import PhysLean.StatisticalMechanics.SpinGlasses.Mathematics.LinearAlgebra.Matrix.PerronFrobenius.Dominance

namespace Matrix
open Finset CollatzWielandt LinearMap

variable {n : Type*} [Fintype n] [Nonempty n] [DecidableEq n]
variable {A : Matrix n n ℝ}

/--
Helper lemma: If ker(f^2) = ker(f), then ker(f^k) = ker(f) for all k ≥ 1.
This shows that the ascent of the kernel stabilizes at 1.
-/
lemma LinearMap.ker_pow_eq_ker_of_ker_sq_eq_ker
    {R M : Type*} [Semiring R] [AddCommMonoid M] [Module R M]
    (f : M →ₗ[R] M) (h_stable : LinearMap.ker (f^2) = LinearMap.ker f) :
    ∀ k ≥ 1, LinearMap.ker (f^k) = LinearMap.ker f := by
  intro k hk
  -- reduce k ≥ 1 to k = m + 1
  obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.one_le_iff_ne_zero.mp hk)
  -- prove ∀ m, ker(f^(m+1)) = ker f by induction on m
  induction' m with m ih
  · -- m = 0: ker(f^1) = ker f
    simp [pow_one]
  · -- step: from ker(f^(m+1)) = ker f show ker(f^(m+2)) = ker f
    apply le_antisymm
    · -- ker(f^(m+2)) ⊆ ker f
      intro x hx
      -- (f^(m+2)) x = 0 ⇒ (f^(m+1)) (f x) = 0
      have hx' : (f ^ (m + 1)) (f x) = 0 := by
        aesop
      -- so f x ∈ ker(f^(m+1)) = ker f, hence f x = 0
      have : f x ∈ LinearMap.ker (f ^ (m + 1)) := by
        simpa [LinearMap.mem_ker] using hx'
      have : f x ∈ LinearMap.ker f := by simpa [ih] using this
      rw [← h_stable]
      simpa [LinearMap.mem_ker] using this
    · -- ker f ⊆ ker(f^(m+2))
      intro x hx
      -- if f x = 0 then (f^(m+2)) x = (f^(m+1)) (f x) = 0
      have : (f ^ (m + 1)) (f x) = 0 := by simp_all
      simpa [pow_succ] using this
/--
The geometric multiplicity of the Perron root of an irreducible non-negative matrix is 1.
The eigenspace is spanned by the unique positive eigenvector.
-/
lemma geometric_multiplicity_one_of_irreducible
    (hA_irred : Irreducible A) (hA_nonneg : ∀ i j, 0 ≤ A i j) :
    let r := perronRoot_alt A
    ∃ v : n → ℝ, (∀ i, 0 < v i) ∧ Module.End.eigenspace (toLin' A) r = Submodule.span ℝ {v} := by
  let r := perronRoot_alt A
  let f := toLin' A

  -- Existence of the positive eigenvector v.
  obtain ⟨r_ex, v, hr_pos, hv_pos, hv_eig_mat, hr_eq_r⟩ := perron_root_eq_positive_eigenvalue hA_irred hA_nonneg
  rw [← hr_eq_r] at hv_eig_mat hr_pos
  have hv_eig_f : f v = r • v := by rwa [toLin'_apply]

  use v, hv_pos

  -- We show E_r = span {v}.
  apply Submodule.ext
  intro w
  constructor
  · -- E_r ⊆ span {v}. Let w ∈ E_r.
    intro hw_E_r
    have hw_eig : f w = r • w := by
      -- unfold f and r so the membership matches exactly
      simpa [f, r] using (Module.End.mem_eigenspace_iff.mp (by assumption))

    -- Split on w = 0.
    by_cases hw_zero : w = 0
    · -- Zero vector lies in the span.
      subst hw_zero
      exact Submodule.zero_mem _
    · -- Continue under w ≠ 0.

      -- Consider |w|.
      let w_abs := fun i => |w i|
      have hw_abs_nonneg : ∀ i, 0 ≤ w_abs i := fun i => abs_nonneg _
      have hw_abs_ne_zero : w_abs ≠ 0 := by
        contrapose! hw_zero
        ext i
        exact abs_eq_zero.mp (congr_fun hw_zero i)

      -- Show r • |w| ≤ f |w| (Subinvariance).
      have h_subinv : r • w_abs ≤ f w_abs := by
        intro i
        classical
        calc
          (r • w_abs) i
              = r * |w i| := by simp [w_abs]
          _ = |r * w i| := by rw [abs_mul, abs_of_pos hr_pos]
          _ = |(f w) i| := by rw [hw_eig]; simp
          _ = |∑ j, A i j * w j| := by simp [f, toLin'_apply, mulVec_apply]
          _ ≤ ∑ j, |A i j * w j| := by
            -- triangle inequality for finite sums
            simpa using
              (Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset n))
                (f := fun j => A i j * w j))
          _ = ∑ j, A i j * |w j| := by
            simp_rw [abs_mul, abs_of_nonneg (hA_nonneg i _)]
          _ = (f w_abs) i := by simp [f, toLin'_apply, mulVec_apply, w_abs]

      -- Show f |w| = r • |w| (Equality upgrade).
      have hw_abs_eig : f w_abs = r • w_abs :=
        subinvariant_equality_implies_eigenvector hA_irred hA_nonneg hw_abs_nonneg hw_abs_ne_zero h_subinv

      -- |w| is strictly positive.
      have hw_abs_pos : ∀ i, 0 < w_abs i :=
          eigenvector_is_positive_of_irreducible hA_irred hw_abs_eig hw_abs_nonneg hw_abs_ne_zero

      -- |w| is a multiple of v.
      obtain ⟨c_abs, hc_abs_pos, hc_abs_eq⟩ :=
        uniqueness_of_positive_eigenvector_gen hA_irred hr_pos hw_abs_pos hv_pos hw_abs_eig hv_eig_f

      -- Now we show w is sign-aligned. w = s * |w| where s ∈ {1, -1}.
      -- We use the standard trick: analyze the primitive matrix B = I + A.

      let B := 1 + A
      -- B properties (non-neg, primitive)
      have hB_nonneg : ∀ i j, 0 ≤ B i j := by
         intro i j; by_cases h : i = j
         · subst h; have := hA_nonneg i i; simp [B]; linarith
         · simp [B, h, hA_nonneg i j]
      have hB_irred := hA_irred.add_one
      have hB_diag_pos : ∀ i, 0 < B i i := by
        intro i; have := hA_nonneg i i; simp [B]; linarith
      have hB_prim : IsPrimitive B := IsPrimitive.of_irreducible_pos_diagonal B hB_nonneg hB_irred hB_diag_pos

      let rB := r + 1
      have hrB_pos : 0 < rB := by linarith [hr_pos]

      -- w is an eigenvector of B for rB.
      have hBw_eig : toLin' B w = rB • w := by
        simp [B, LinearMap.add_apply, toLin'_one, add_smul, one_smul, rB]; abel_nf; aesop

      -- |w| is an eigenvector of B for rB.
      have hw_abs_eig_B : toLin' B w_abs = rB • w_abs := by
        simp [B, LinearMap.add_apply, toLin'_one, add_smul, one_smul, rB]; abel_nf; aesop

      -- We view w as a complex vector wc to apply the phase alignment results for primitive matrices.
      let wc : n → ℂ := fun i => (w i : ℂ)
      have hwc_eig_B : (B.map (algebraMap ℝ ℂ)) *ᵥ wc = (rB : ℂ) • wc := by
        classical
        ext i
        -- First, record the real componentwise equality at index i.
        have h_real : ∑ j, B i j * w j = rB * w i := by
          have := congrArg (fun v : n → ℝ => v i)
            (by simpa [toLin'_apply, Pi.smul_apply] using hBw_eig)
          simpa [Matrix.mulVec, dotProduct] using this
        -- Lift to ℂ and compute both sides componentwise.
        calc
          ((B.map (algebraMap ℝ ℂ)) *ᵥ wc) i
              = ∑ j, (algebraMap ℝ ℂ (B i j)) * wc j := by
                simp [Matrix.mulVec, dotProduct]
          _ = ∑ j, (B i j : ℂ) * (w j : ℂ) := by
                simp [wc]
          _ = ∑ j, Complex.ofReal (B i j * w j) := by
                simp [Complex.ofReal_mul]
          _ = Complex.ofReal (∑ j, B i j * w j) := by
                simp
          _ = Complex.ofReal (rB * w i) := by
                simp [h_real]
          _ = (rB : ℂ) * wc i := by
                simp [wc, Complex.ofReal_mul]

      -- We need the Perron root of B.
      have hrB_eq_perronB : rB = perronRoot_alt B :=
        eigenvalue_is_perron_root_of_positive_eigenvector hB_irred hB_nonneg hrB_pos hw_abs_pos hw_abs_eig_B

      -- Apply phase alignment theorem for primitive matrices.
      have h_norm_eig : (B *ᵥ fun i => ‖wc i‖) = perronRoot_alt B • (fun i => ‖wc i‖) := by
        have h_eq : (fun i => ‖wc i‖) = w_abs := by
          ext i
          simp [wc, w_abs]
        simpa [toLin'_apply, hrB_eq_perronB, h_eq] using hw_abs_eig_B
      have h_norm_pos : ∀ i, 0 < ‖wc i‖ := by
        intro i
        simpa [wc, w_abs, Complex.norm_ofReal] using hw_abs_pos i
      obtain ⟨c, hc_norm, hc_eq⟩ := eigenvector_phase_aligned_of_primitive hB_prim hB_nonneg
        (by simpa [hrB_eq_perronB] using perronRoot_nonneg hB_nonneg)
        hwc_eig_B
        h_norm_eig
        h_norm_pos

      -- c must be real (c = ±1) because w is real.
      let i := Classical.arbitrary n
      have hc_eq_i := congr_fun hc_eq i
      simp only [wc] at hc_eq_i
      -- (w i : ℂ) = c * (|w i| : ℂ).

      have hc_real : c.im = 0 := by
        -- Take imaginary parts in the scalar alignment equality at index i.
        have h := congrArg Complex.im hc_eq_i
        -- Compute the imaginary part (|w i| is real and positive).
        have h' : c.im * w_abs i = 0 := by
          -- (w i : ℂ).im = 0 and (↑(w_abs i) : ℂ) is real.
          have : 0 = (c * (w_abs i : ℂ)).im := by
            simpa [wc, w_abs, Complex.ofReal_mul, Complex.ofReal_im, Complex.ofReal_re] using h.symm
          simpa [Complex.mul_im] using this
        exact (mul_eq_zero.mp h').resolve_right (ne_of_gt (hw_abs_pos i))

      -- w = c.re • w_abs.
      have hw_eq_c_wabs : w = c.re • w_abs := by
        ext j
        -- Take real parts in the alignment equality at index j.
        have h := congrArg Complex.re (congr_fun hc_eq j)
        -- Compute real parts; note that (↑‖(w j : ℂ)‖ : ℂ) is real.
        have : w j = c.re * ‖wc j‖ := by
          simpa [wc, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im] using h
        -- Replace ‖wc j‖ with |w j| and rewrite pointwise scalar multiplication.
        simpa [Pi.smul_apply, w_abs, wc, Complex.norm_ofReal, smul_eq_mul] using this

      -- Combine results: w = c.re • (c_abs • v) = (c.re * c_abs) • v.
      rw [hw_eq_c_wabs, hc_abs_eq, smul_smul]
      exact Submodule.mem_span_singleton.mpr ⟨c.re * c_abs, rfl⟩

  · -- span {v} ⊆ E_r.
    intro hw
    rcases Submodule.mem_span_singleton.mp hw with ⟨c, rfl⟩
    -- show f (c • v) = r • (c • v)
    have hc : f (c • v) = r • (c • v) := by
      calc
        f (c • v) = c • f v := by simp
        _ = c • (r • v) := by simp [hv_eig_f]
        _ = (c * r) • v := by simp [smul_smul]
        _ = (r * c) • v := by simp [mul_comm]
        _ = r • (c • v) := by simp [smul_smul]
    -- close the eigenspace membership
    have : (toLin' A) (c • v) = (perronRoot_alt A) • (c • v) := by
      simpa [f, r]
        using hc
    exact (Module.End.mem_eigenspace_iff).2 this


open scoped Matrix InnerProductSpace

/--
The algebraic multiplicity of the Perron root of an irreducible non-negative matrix is 1.
The generalized eigenspace equals the eigenspace.
-/
lemma algebraic_multiplicity_one_of_irreducible
    (hA_irred : Irreducible A) (hA_nonneg : ∀ i j, 0 ≤ A i j) :
    Module.End.maxGenEigenspace (toLin' A) (perronRoot_alt A) =
    Module.End.eigenspace (toLin' A) (perronRoot_alt A) := by
  classical
  let r := perronRoot_alt A
  let f := toLin' A
  -- Define g = f - rI.
  let g := f - r • LinearMap.id

  -- We show ker(g^2) = ker(g).
  have h_ker_sq_eq_ker : LinearMap.ker (g ^ 2) = LinearMap.ker g := by
    apply le_antisymm
    · -- ker(g^2) ⊆ ker(g).
      intro w hw_g2
      -- Let u = g(w). Then g(u) = g^2(w) = 0.
      let u := g w
      have hu_g : g u = 0 := by
        -- g u = g (g w) = (g^2) w = 0
        simpa [u, pow_two, LinearMap.comp_apply] using hw_g2
      -- u is an eigenvector of f for r.
      have hu_eig : f u = r • u := by
        -- g u = 0 ⇒ (f - r•id) u = 0 ⇒ f u = r • u
        simp only [g, LinearMap.sub_apply, sub_eq_zero] at hu_g
        exact hu_g
      -- Use a positive left eigenvector u* of Aᵀ.
      have hAT_irred := hA_irred.transpose hA_nonneg
      obtain ⟨rT, u_star, hrT_pos, hu_star_pos, hu_star_eig_T_mat⟩ :=
        exists_positive_eigenvector_of_irreducible hAT_irred
      -- The eigenvalue rT equals r (transpose preserves Perron root).
      have hrT_eq_r : rT = r := by
        calc
          rT = perronRoot_alt Aᵀ :=
                eigenvalue_is_perron_root_of_positive_eigenvector
                  hAT_irred (fun i j => hA_nonneg j i) hrT_pos hu_star_pos hu_star_eig_T_mat
          _  = r := (perronRoot_transpose_eq A hA_irred).symm
      -- Rewrite u_star's eigenvector equation with r
      have hu_star_eig_r : (toLin' Aᵀ) u_star = r • u_star := by
        simpa [hrT_eq_r, toLin'_apply] using congrArg id hu_star_eig_T_mat
      -- Compute u_star ⬝ (g w) = 0
      have h_dot_g_w : u_star ⬝ᵥ (g w) = 0 := by
        calc
          u_star ⬝ᵥ (g w)
              = u_star ⬝ᵥ (f w) - u_star ⬝ᵥ (r • w) := by
                  simp [g, dotProduct_sub]
          _ = (toLin' Aᵀ u_star) ⬝ᵥ w - r * (u_star ⬝ᵥ w) := by
                -- u_star ⬝ (f w) = (toLin' Aᵀ u_star) ⬝ w
                -- and u_star ⬝ (r • w) = r * (u_star ⬝ w)
                have h₁ : u_star ⬝ᵥ (f w) = (toLin' Aᵀ u_star) ⬝ᵥ w := by
                  -- Standard adjoint identity via dotProduct–mulVec
                  simp [f, toLin'_apply]; exact dotProduct_mulVec_comm u_star w A
                simp [h₁, dotProduct_smul, smul_eq_mul]
          _ = (r • u_star) ⬝ᵥ w - r * (u_star ⬝ᵥ w) := by
                simp [hu_star_eig_r]
          _ = r * (u_star ⬝ᵥ w) - r * (u_star ⬝ᵥ w) := by
                simp [smul_eq_mul]
          _ = 0 := by ring
      -- Since u = g(w), we have u* ⬝ u = 0.
      have h_dot_u : u_star ⬝ᵥ u = 0 := by simpa [u] using h_dot_g_w
      -- Geometric multiplicity 1: eigenspace is span of a positive vector.
      obtain ⟨v, hv_pos, h_Er_span⟩ :=
        geometric_multiplicity_one_of_irreducible (A := A) hA_irred hA_nonneg
      have hu_in_Er : u ∈ Module.End.eigenspace f r := by
        -- membership from hu_eig
        simpa [Module.End.mem_eigenspace_iff, f, r] using hu_eig
      -- project u onto the 1-dim span
      have : u ∈ Submodule.span ℝ ({v} : Set (n → ℝ)) := by
        simpa [h_Er_span, r, f] using hu_in_Er
      obtain ⟨c, hc_eq⟩ := Submodule.mem_span_singleton.mp this
      -- Substitute back into the dot product: c * (u* ⬝ v) = 0, hence c = 0 or u* ⬝ v = 0.
      have h_dot_u' : c = 0 ∨ u_star ⬝ᵥ v = 0 := by
        -- u_star ⬝ (c • v) = c * (u_star ⬝ v) and u = c • v
        have hc_mul : c * (u_star ⬝ᵥ v) = 0 := by
          -- first rewrite u as c • v in the dot product
          have : u_star ⬝ᵥ (c • v) = 0 := by
            simpa [hc_eq] using h_dot_u
          -- then use linearity on the right to pull out the scalar
          simpa [dotProduct_smul, smul_eq_mul, mul_comm] using this
        exact mul_eq_zero.mp hc_mul
      -- u* ⬝ v > 0 because both are positive vectors.
      have hv_ne_zero : v ≠ 0 := by
        intro h
        have : 0 < v (Classical.arbitrary n) := hv_pos _
        have : False := by simp [h] at this
        exact this.elim
      have h_dot_pos : 0 < u_star ⬝ᵥ v :=
        dotProduct_pos_of_pos_of_nonneg_ne_zero hu_star_pos
          (fun i => (hv_pos i).le) hv_ne_zero
      -- Therefore c = 0.
      have hc_zero : c = 0 := h_dot_u'.resolve_right h_dot_pos.ne'
      -- Hence u = 0.
      have hu_zero : u = 0 := by
        simpa [hc_zero, zero_smul] using hc_eq.symm
      -- Then g(w) = 0, so w ∈ ker(g).
      -- Convert to kernel membership.
      have : g w = 0 := by simpa [u] using hu_zero
      simpa [LinearMap.mem_ker] using this
    · -- ker(g) ⊆ ker(g^2). This is always true.
      intro x hx
      have hx0 : g x = 0 := by simpa [LinearMap.mem_ker] using hx
      have : (g ^ 2) x = 0 := by
        simp [pow_two, hx0]
      simpa [LinearMap.mem_ker] using this

  -- Since the kernel stabilizes at k=1 (by h_ker_sq_eq_ker), the generalized eigenspace equals the eigenspace.
  haveI : FiniteDimensional ℝ (n → ℝ) := by infer_instance

  -- Unfold maxGenEigenspace to an iSup of kernels of powers, and show it stabilizes at 1.
  -- First, show ker(g^k) = ker(g) for all k ≥ 1.
  have h_stabilize := LinearMap.ker_pow_eq_ker_of_ker_sq_eq_ker g h_ker_sq_eq_ker

  -- The supremum over k stabilizes at k = 1.
  -- We use the definition of maxGenEigenspace and simplify.
  -- maxGenEigenspace (f,r) = ⨆ k, ker ((f - r•id)^k) = ⨆ k, ker (g^k).
  -- Then we show the iSup equals ker g.
  have h_sup_eq : ⨆ (k : ℕ), LinearMap.ker (g ^ k) = LinearMap.ker g := by
    apply le_antisymm
    · -- ⨆ k, ker(g^k) ≤ ker g
      apply iSup_le
      intro k
      cases k with
      | zero =>
        -- ker(g^0) = ker(id) = ⊥ ≤ ker g
        intro x hx
        have hx0 : x = 0 := by
          simpa [pow_zero, LinearMap.mem_ker] using hx
        simp [hx0]
      | succ k' =>
        -- ker(g^(k'+1)) = ker g by stabilization
        have hk' : 1 ≤ k'.succ := Nat.succ_le_succ (Nat.zero_le _)
        simp [h_stabilize k'.succ hk']
    · -- ker g ≤ ⨆ k, ker(g^k), by picking k=1
      exact le_iSup_of_le 1 (by simp [pow_one])

  -- Conclude:
  -- maxGenEigenspace (toLin' A) r = ⨆ k, ker(g^k) = ker g = eigenspace f r.
  -- The last equality uses the definition of eigenspace.
  calc
    Module.End.maxGenEigenspace f r
      = ⨆ k, LinearMap.ker (g ^ k) := by
        simp [Module.End.maxGenEigenspace, Module.End.genEigenspace, g]; rfl
    _ = LinearMap.ker g := h_sup_eq
    _ = Module.End.eigenspace f r := by
        simp [Module.End.eigenspace, g]; rw [@Module.End.genEigenspace_one]; rfl

end Matrix
