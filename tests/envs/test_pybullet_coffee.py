"""Test the PyBullet Coffee environment."""

import pytest

from predicators import utils
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv


@pytest.fixture(scope="module", name="env")
def _create_exposed_pybullet_coffee_env(request):
    utils.reset_config({
        "env": "pybullet_coffee",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "coffee_num_cups_train": [1],
        "coffee_num_cups_test": [1],
    })
    env = PyBulletCoffeeEnv(use_gui=False)
    yield env


def test_pybullet_coffee_basic(env):
    """Test basic environment functionality."""
    assert env.get_name() == "pybullet_coffee"
    
    # Test reset
    env.reset("train", 0)
    
    # Get initial state
    state = env._current_observation
    assert state is not None
    
    # Check that all expected objects exist
    assert env._robot in state
    assert env._jug in state
    assert env._machine in state
    
    cups = state.get_objects(env._cup_type)
    assert len(cups) == 1  # We set coffee_num_cups_train to [1]
    
    # Test action space
    action_space = env.action_space
    assert action_space.shape == (8,)  # PyBullet robot has 8-dim action space
    
    # Test step with a dummy action
    action = env.action_space.sample()
    action_arr = utils.Action(action)
    next_state = env.step(action_arr)
    assert next_state is not None
    
    print("Basic PyBullet Coffee environment test passed!")


def test_pybullet_coffee_predicates(env):
    """Test that predicates work correctly."""
    env.reset("train", 0)
    state = env._current_observation
    
    # Test HandEmpty predicate
    hand_empty = env._HandEmpty_holds(state, [env._robot])
    assert hand_empty  # Initially, hand should be empty
    
    # Test OnTable predicate
    on_table = env._OnTable_holds(state, [env._jug])
    assert on_table  # Initially, jug should be on table
    
    # Test MachineOn predicate
    machine_on = env._MachineOn_holds(state, [env._machine])
    assert not machine_on  # Initially, machine should be off
    
    # Test JugFilled predicate
    jug_filled = env._JugFilled_holds(state, [env._jug])
    assert not jug_filled  # Initially, jug should be empty
    
    print("Predicate tests passed!")


if __name__ == "__main__":
    # Run tests manually
    utils.reset_config({
        "env": "pybullet_coffee",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "coffee_num_cups_train": [1],
        "coffee_num_cups_test": [1],
    })
    env = PyBulletCoffeeEnv(use_gui=False)
    
    test_pybullet_coffee_basic(env)
    test_pybullet_coffee_predicates(env)
    
    print("All tests passed!")
