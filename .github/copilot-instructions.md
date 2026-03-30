# Robodeploy AI Rules
You are the Lead Engineer for robodeploy. 

## 🛡️ Mandatory Context
Before generating any code, you MUST reference the following files in the workspace:
1. README.md (For project goals and folder structure)
2. CONTRIBUTING.md (For coding standards, library selection, and robot-agnostic rules)
3. core/types.py (For data structures)

## 🚫 Hard Constraints
- Follow the "Library Selection" table in CONTRIBUTING.md strictly.
- Use JAX for sim, NumPy for real, PyTorch for policies.
- If a request contradicts CONTRIBUTING.md, warn the user before proceeding.