# Harry Potter: Stochastic Wizard Planning

This project tackles a stochastic decision-making problem set in the world of Harry Potter. Wizards attempt to destroy horcruxes while avoiding unpredictable death eaters — all under a strict turn limit. The challenge involves intelligent planning under uncertainty using value iteration, heuristics, and optimal control.

## 🧙‍♂️ Project Overview
You control one or more wizards navigating a grid-world. The goals:
- Destroy horcruxes to earn points.
- Avoid encounters with randomly-moving death eaters (each encounter costs points).
- Strategically reset or terminate the environment when beneficial.

The environment includes:
- **Horcruxes** that may randomly relocate.
- **Death Eaters** that move stochastically on predefined paths.
- **Actions**: move, wait, destroy horcrux, reset environment, terminate game.

## 📂 Repository Contents
- `ex3.py`: Your main implementation file. Contains both `WizardAgent` (approximate policy) and `OptimalWizardAgent` (value-iteration-based).
- `check.py`: Simulator and scorer — runs the environment and evaluates your agent.
- `inputs.py`: Predefined testing environments.
- `utils.py`: Utility functions used throughout the project.
- `HW3.pdf`: Full problem description and constraints.

## 🧠 Agents
- `WizardAgent`: Uses partial BFS, heuristic fallback, and value iteration (under time constraints).
- `OptimalWizardAgent`: Builds and evaluates the full state space to find the optimal policy (within limits).

## 🚀 Run the Project
```bash
python check.py
