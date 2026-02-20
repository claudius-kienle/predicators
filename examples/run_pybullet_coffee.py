"""Example script demonstrating the PyBullet Coffee environment.

Usage:
    python examples/run_pybullet_coffee.py
    
This script shows how to:
1. Create and initialize the PyBullet Coffee environment
2. Reset the environment and get initial state
3. Execute random actions and observe state changes
4. Check predicates and goal conditions
"""

from predicators import utils
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv


def main():
    """Run a simple demo of the PyBullet Coffee environment."""
    
    # Configure the environment
    utils.reset_config({
        "env": "pybullet_coffee",
        "num_train_tasks": 3,
        "num_test_tasks": 2,
        "coffee_num_cups_train": [2, 2],
        "coffee_num_cups_test": [2, 3],
        "pybullet_robot": "panda",  # or "fetch"
    })
    
    # Create the environment with GUI enabled
    print("Creating PyBullet Coffee environment...")
    env = PyBulletCoffeeEnv(use_gui=True)
    
    print(f"Environment name: {env.get_name()}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space bounds: {env.action_space.low} to {env.action_space.high}")
    
    # Reset to a training task
    print("\nResetting to training task 0...")
    state = env.reset("train", 0)
    
    # Print initial state information
    print(f"\nInitial state:")
    print(f"  Robot position: ({state.get(env._robot, 'x'):.2f}, "
          f"{state.get(env._robot, 'y'):.2f}, "
          f"{state.get(env._robot, 'z'):.2f})")
    print(f"  Jug position: ({state.get(env._jug, 'x'):.2f}, "
          f"{state.get(env._jug, 'y'):.2f})")
    print(f"  Jug rotation: {state.get(env._jug, 'rot'):.2f}")
    print(f"  Machine on: {state.get(env._machine, 'is_on') > 0.5}")
    print(f"  Jug filled: {state.get(env._jug, 'is_filled') > 0.5}")
    
    # Get cups and their states
    cups = state.get_objects(env._cup_type)
    print(f"\nNumber of cups: {len(cups)}")
    for i, cup in enumerate(cups):
        print(f"  Cup {i}: position ({state.get(cup, 'x'):.2f}, "
              f"{state.get(cup, 'y'):.2f}), "
              f"capacity {state.get(cup, 'capacity_liquid'):.2f}, "
              f"current {state.get(cup, 'current_liquid'):.2f}")
    
    # Check initial predicates
    print("\nInitial predicates:")
    print(f"  HandEmpty: {env._HandEmpty_holds(state, [env._robot])}")
    print(f"  OnTable(jug): {env._OnTable_holds(state, [env._jug])}")
    print(f"  Holding(robot, jug): {env._Holding_holds(state, [env._robot, env._jug])}")
    print(f"  MachineOn: {env._MachineOn_holds(state, [env._machine])}")
    print(f"  JugFilled: {env._JugFilled_holds(state, [env._jug])}")
    
    # Print goal
    task = env.get_task("train", 0)
    print(f"\nGoal: {task.goal}")
    
    # Execute some random actions
    print("\nExecuting 10 random actions...")
    for step in range(10):
        action = env.action_space.sample()
        action_obj = utils.Action(action)
        state = env.step(action_obj)
        
        if step % 5 == 0:
            print(f"  Step {step}: Robot at ({state.get(env._robot, 'x'):.2f}, "
                  f"{state.get(env._robot, 'y'):.2f}, "
                  f"{state.get(env._robot, 'z'):.2f})")
    
    print("\nDemo complete! Close the PyBullet window to exit.")
    
    # Keep the window open
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
