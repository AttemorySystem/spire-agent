# Small SubAgent boundary

All three owners use the same fixed pipeline:

```text
continuation stage -> fast path -> tool stage -> optional fallback
```

- A continuation stage resumes only an owner-bound multi-screen operation.
- A fast path contains a small deterministic screen rule.
- A tool stage translates between `DecisionRequest` and a narrow domain tool;
  MCTS and Winning Path belong behind such stages, not inside the pipeline.
- A fallback is the only stage expected to call an LLM.

Every stage only reads the immutable request and returns `Decision` or `None`.
It cannot execute commands or mutate context. Exceptions propagate, and an
unhandled request fails closed instead of guessing an action.

Provider-neutral LLM request/response values and BuildAgent's room-scoped
context live here because their contracts are owned by the consuming agent.
Provider calls belong in `src/spire_agent/adapters/`; recording and replay belong in
`src/spire_agent/extensions/`; concrete Map, MCTS, and Winning Path implementations
belong in `src/spire_agent/tools/`.
