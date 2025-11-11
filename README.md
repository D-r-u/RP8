### Check for Albation
- in marl_agents/self_maddpg.py:
- def _get_coordination_signals(self, states, mode="soft"):
- try for mode : ["soft", "diag0", "diag1", "binary"]

### Try for different Rewards
- in environment/wmn_env.py:
- def _calculate_reward(self, states, mode="balanced"):
- try for mode : ["balanced", "gaussian"]