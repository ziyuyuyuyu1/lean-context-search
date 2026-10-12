/- Core Lean fixtures appended to the exporter in a generated smoke wrapper. -/
namespace ContextFixtures
universe u

def Fixes {α : Type u} (f : α → α) : Prop := ∀ x, f x = x

structure Parent (α : Type u) where
  payload : α
  valid : payload = payload

structure Child (α : Type u) extends Parent (List α) where
  extra : α

structure Left extends Parent Nat where
  left : Nat

structure Right extends Parent Nat where
  right : Nat

structure Diamond extends Left, Right where
  extra : Nat

class Cap (α : Type u) where
  op : α → α

class Rich (α : Type u) extends Cap α where
  law : Fixes op

structure Dependent where
  carrier : Type u
  point : carrier
  predicate : carrier → Prop
  law : predicate point

theorem ordinary : True := True.intro
end ContextFixtures
