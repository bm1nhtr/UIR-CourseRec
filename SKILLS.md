1. Plan mode by default.
Enter plan mode for any task with 3+ steps or architectural decisions. If something goes sideways, stop and re-plan immediately. Write detailed specs upfront to reduce ambiguity. Planning is how you stay in control.

2. Use subagents liberally.
Offload research, exploration, and parallel analysis to subagents. Keep your main context window clean. For complex problems, throw more compute at it. One task per subagent for focused execution.

3. Build a self-improvement loop.
After any correction, update a lessons file with the pattern. Write rules that prevent the same mistake. Ruthlessly iterate on these lessons until mistake rates drop. Review them at the start of every session.

4. Verify before marking done.
Never mark a task complete without proving it works. Diff behavior between main and your changes. Ask yourself: would a staff engineer approve this? Run tests, check logs, demonstrate correctness.

5. Demand elegance, but stay balanced.
For non-trivial changes, pause and ask if there's a more elegant way. If a fix feels hacky, implement the elegant solution. But skip this for simple, obvious fixes. Challenge your own work before presenting it.

6. Let AI fix bugs autonomously.
When given a bug report, just fix it. Point at logs, errors, failing tests and resolve them. Go fix failing CI tests without being told how.

7. Keep repo truth in sync.
README tree, script names, and lockfiles must match what is on disk (no phantom entry points). After changing dependencies, run `poetry lock` when PyPI is reachable and keep `requirements.txt` aligned with `pyproject.toml` if pip-based setup scripts still read it. Also scan docstrings and comments inside `.py` files—documentation drift often hides there, not only in README.