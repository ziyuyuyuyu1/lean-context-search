# Context examples

Shortened views of five measured passages. Notation is simplified; these are illustrations, not the exact strings encoded by the models.

## Monoid

**Before:** a description of a monoid and the signature `Monoid : Type u → Type u`.

**Added context — excerpt:**

```text
Monoid M
  parents: Semigroup M, MulOneClass M
  mul : M → M → M
  one : M
  mul_assoc : (a * b) * c = a * (b * c)
  one_mul   : 1 * a = a
  mul_one   : a * 1 = a
```

Makes the inherited operations and their laws explicit.

## Equiv

**Before:** a description of a bijection and the type of equivalences between `α` and `β`.

**Added context:**

```text
Equiv α β
  toFun  : α → β
  invFun : β → α
  left_inv  : invFun (toFun x) = x
  right_inv : toFun (invFun y) = y
```

Shows the two maps and the laws that make them inverses.

## Filter

**Before:** an informal description and the signature `Filter : Type u → Type u`.

**Added context:**

```text
Filter α
  sets : Set (Set α)
  univ_sets        : univ ∈ sets
  sets_of_superset : s ∈ sets → s ⊆ t → t ∈ sets
  inter_sets       : s ∈ sets → t ∈ sets → s ∩ t ∈ sets
```

Makes the required closure laws explicit. A separate nontriviality condition is needed to exclude the empty set.

## LinearMap

**Before:** a description of an additive map between modules that respects a ring homomorphism `σ : R →+* S`.

**Added context:**

```text
LinearMap σ M M₂
  parents: AddHom M M₂, MulActionHom (⇑σ) M M₂
  toFun : M → M₂
  map_add'  : toFun (x + y) = toFun x + toFun y
  map_smul' : toFun (r • x) = σ(r) • toFun x
```

Preserves the parameter substitution: scalar multiplication on the target uses `σ(r)`.

## LinearEquiv

**Before:** a description of a bijective map between modules compatible with inverse ring homomorphisms.

**Added context:**

```text
LinearEquiv σ M M₂
  parents: LinearMap σ M M₂, AddEquiv M M₂
  toFun  : M → M₂
  invFun : M₂ → M
  map_add'  : toFun (x + y) = toFun x + toFun y
  map_smul' : toFun (r • x) = σ(r) • toFun x
  left_inv  : invFun (toFun x) = x
  right_inv : toFun (invFun y) = y
```

Combines the inherited linear-map laws with the inverse-map laws.
