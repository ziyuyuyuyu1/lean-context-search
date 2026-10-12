import Lean

/-
Template: complete_context.py inserts pinned imports after `import Lean` and
appends `export_structure_context "names.txt" "contexts.jsonl"`.
No declarations or proof bodies are changed. The export is a textual interface,
not a compilable replacement structure declaration.
-/
open Lean Meta Elab

namespace CompleteContext

def js (s : String) : Json := Json.str s
def strings (xs : Array String) : Json := Json.arr (xs.map js)

def pp (e : Expr) : MetaM String := do
  withOptions (fun o => o.set `pp.width 120 |>.setBool `pp.universes true
    |>.setBool `pp.fullNames true |>.setBool `pp.proofs true
    |>.set `pp.maxDepth 100000 |>.set `pp.maxSteps 1000000) do
    let text := (← ppExpr e).pretty
    if text.contains '⋯' then throwError "pretty-printer elided part of the interface"
    return text

def closedType (context : Array Expr) (type : Expr) : MetaM String := do
  let closed ← mkForallFVars context type (usedOnly := false)
  unless !closed.hasFVar && !closed.hasMVar && !closed.hasLooseBVars do
    throwError "exported type contains free/metavariables"
  check closed
  withOptions (fun o => o.setBool `pp.explicit true |>.setBool `pp.notation false
    |>.setBool `pp.fieldNotation false) do
    pp closed

/- Traverse actual stored subobjects. A repeated leaf name is deduplicated only
after checking that its accessors agree definitionally, not just textually. -/
partial def leafAccessors (self : Expr) : MetaM (Array (Name × Name × Expr)) := do
  let env ← getEnv
  let type ← whnf (← inferType self)
  let .const owner _ := type.getAppFn | throwError "expected a structure application"
  let mut out := #[]
  for field in getStructureFields env owner do
    let value ← mkProjection self field
    if (isSubobjectField? env owner field).isSome then
      out := out ++ (← leafAccessors value)
    else
      out := out.push (field, owner, value)
  return out

def describe (name : Name) : MetaM Json := do
  let env ← getEnv
  let ind ← getConstInfoInduct name
  unless ind.numIndices == 0 do
    throwError "indexed structure requires a separate extractor"
  let ctor := getStructureCtor env name
  let constructorType ← pp ctor.type
  let declarationType ← pp ind.type
  let doc ← findDocString? env name
  forallBoundedTelescope ind.type ind.numParams (cleanupAnnotations := false) fun params _ => do
    let selfType := mkAppN (mkConst name (ind.levelParams.map mkLevelParam)) params
    let selfBI := if isClass env name then BinderInfo.instImplicit else BinderInfo.default
    withLocalDecl `self selfBI selfType fun self => do
      let context := params.push self
      let mut parameterRecords := #[]
      for p in params do
        let d ← p.fvarId!.getDecl
        parameterRecords := parameterRecords.push <| Json.mkObj [
          ("name", js d.userName.toString), ("binder_info", js (reprStr d.binderInfo)),
          ("type", js (← pp d.type))]
      let mut parents := #[]
      for parent in getStructureParentInfo env name do
        let value ← mkAppOptM parent.projFn ((params.map some).push (some self))
        check value
        unless !value.hasMVar do throwError "unresolved parent projection"
        let type ← inferType value
        parents := parents.push <| Json.mkObj [
          ("name", js parent.structName.toString), ("projection", js parent.projFn.toString),
          ("embedded_subobject", toJson parent.subobject),
          ("application", js (← pp value)), ("type", js (← pp type)),
          ("closed_type", js (← closedType context type))]
      let mut fields := #[]
      let mut seen : Std.HashMap Name Expr := {}
      let leaves ← leafAccessors self
      unless leaves.map (fun (field, _, _) => field) == getStructureFieldsFlattened env name false do
        throwError "effective-field traversal disagrees with native flattened metadata"
      for (field, owner, value) in leaves do
        if let some earlier := seen.get? field then
          unless ← isDefEq earlier value do
            throwError "distinct inherited fields share the name {field}; qualify before rendering"
        else
          seen := seen.insert field value
          check value
          unless !value.hasMVar do throwError "unresolved field projection"
          let type ← inferType value
          let proofField ← isProp type
          let expandedType ← if proofField then withTransparency .default (whnf type) else pure type
          unless ← isDefEq type expandedType do throwError "field normalization changed its type"
          let some info := getFieldInfo? env owner field
            | throwError "missing field metadata for {owner}.{field}"
          let fieldDoc ← findDocString? env info.projFn
          fields := fields.push <| Json.mkObj [
            ("name", js field.toString), ("storage_owner", js owner.toString),
            ("projection", js info.projFn.toString),
            ("accessor", js (← pp value)), ("type", js (← pp type)),
            ("head_expanded_type", js (← pp expandedType)),
            ("closed_type", js (← closedType context type)),
            ("is_proof", toJson proofField),
            ("docstring", toJson fieldDoc)]
      return Json.mkObj [
        ("record_kind", js "structure_context"), ("schema_version", js "complete-context-v1"),
        ("name", js name.toString), ("kind", js (if isClass env name then "class" else "structure")),
        ("level_parameters", strings (ind.levelParams.toArray.map Name.toString)),
        ("declaration_type", js declarationType), ("parameters", Json.arr parameterRecords),
        ("self_type", js (← pp selfType)),
        ("constructor", Json.mkObj [("name", js ctor.name.toString), ("type", js constructorType)]),
        ("parents", Json.arr parents), ("fields", Json.arr fields),
        ("docstring", toJson doc), ("verification", js "projection-expressions-typechecked")]

elab "export_structure_context " namesPath:str outputPath:str : command => do
  let namesText ← IO.FS.readFile namesPath.getString
  let requested := namesText.splitOn "\n" |>.filter (fun s => !s.isEmpty)
  let mut wanted : Std.HashSet String := {}
  for s in requested do wanted := wanted.insert s
  let env ← getEnv
  let mut resolved : Std.HashMap String Name := {}
  for (n, _) in env.constants.toList do
    if wanted.contains n.toString then resolved := resolved.insert n.toString n
  let mut output : Array String := #[]
  let mut missing := #[]
  let mut count := 0
  let mut errors := 0
  for s in requested do
    match resolved.get? s with
    | none => missing := missing.push s
    | some n =>
      if isStructure env n then
        try
          let result ← Command.liftTermElabM <| describe n
          output := output.push result.compress
          count := count + 1
        catch e =>
          errors := errors + 1
          output := output.push (Json.mkObj [("record_kind", js "extraction_error"),
            ("name", js s), ("error", js (← e.toMessageData.toString))]).compress
  let manifest := Json.mkObj [("record_kind", js "extraction_manifest"),
    ("lean_version", js Lean.versionString), ("requested_count", toJson requested.length),
    ("exported_count", toJson count), ("error_count", toJson errors),
    ("missing_names", strings missing)]
  output := output.push manifest.compress
  IO.FS.writeFile outputPath.getString (String.intercalate "\n" output.toList ++ "\n")
  logInfo m!"Exported {count} interfaces; {errors} errors; {missing.size} unresolved names."

end CompleteContext
