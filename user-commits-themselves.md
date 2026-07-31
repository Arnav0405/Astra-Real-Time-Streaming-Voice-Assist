---
name: user-commits-themselves
description: Never commit — user insists on making all git commits personally
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 91cd2173-8e7c-4cc9-84b6-13fa9e74c004
---

User explicitly instructed: never commit anything yourself; instead FORCE them to commit (give exact commands, state that nothing is committed until they act).

**Why:** they want ownership of repo history in [[astra-project]].

**How to apply:** after any change set, stop before `git add`/`git commit`; end with review + commit instructions for the user. Running `git init` or generating lockfiles is fine.
