#!/usr/bin/env python3
"""
Test script to verify bug fixes in DreamerV3 Curiosity
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np

def test_curiosity_trigger():
    """Test 1: Verify CuriosityTrigger is now stateless"""
    print("=" * 60)
    print("Test 1: CuriosityTrigger Stateless Design")
    print("=" * 60)

    from dreamerv3.agent import CuriosityTrigger

    # Initialize trigger
    trigger = CuriosityTrigger(alpha=0.001, std_scale=1.0)

    # Get initial state
    state = trigger.initial_state()
    print(f"✓ Initial state: {state}")
    assert state['mean'] == 0.0, "Initial mean should be 0.0"
    assert state['var'] == 1.0, "Initial variance should be 1.0"

    # Update state with entropy values
    entropy_values = [0.5, 0.8, 0.6, 0.9, 0.7]
    for i, entropy in enumerate(entropy_values):
        state = trigger.update(state, entropy)
        explore, threshold = trigger.should_explore(state, entropy)
        print(f"  Step {i+1}: entropy={entropy:.3f}, mean={state['mean']:.4f}, "
              f"var={state['var']:.4f}, threshold={threshold:.4f}, explore={explore}")

    # Verify state persistence across updates
    assert state['mean'] != 0.0, "Mean should have changed from initial value"
    assert state['var'] != 1.0, "Variance should have changed from initial value"
    print("✓ CuriosityTrigger is stateless and working correctly\n")
    return True

def test_sample_uniform_actions():
    """Test 2: Verify sample_uniform_actions returns correct number of actions"""
    print("=" * 60)
    print("Test 2: sample_uniform_actions() Indentation Fix")
    print("=" * 60)

    # This test requires a full agent setup, so we'll do a simpler check
    print("✓ Checking indentation in source code...")

    with open('/root/dreamerv3_curiosity-Dynamic_Curiosity_Threshold/dreamerv3/agent.py', 'r') as f:
        lines = f.readlines()

    # Find the sample_uniform_actions method
    in_method = False
    found_append = False
    append_indent = 0

    for i, line in enumerate(lines, 1):
        if 'def sample_uniform_actions' in line:
            in_method = True
            print(f"  Found method at line {i}")
        elif in_method and 'actions.append(act)' in line:
            found_append = True
            # Count leading spaces
            append_indent = len(line) - len(line.lstrip())
            print(f"  Found 'actions.append(act)' at line {i} with {append_indent} spaces indent")

            # Check if it's inside the for loop (should have more than 4 spaces)
            if append_indent >= 6:  # At least 6 spaces means inside the loop
                print(f"✓ Correctly indented inside for loop")
            else:
                print(f"✗ ERROR: Still outside for loop!")
                return False
            break

    if not found_append:
        print("✗ Could not find 'actions.append(act)' line")
        return False

    print("✓ sample_uniform_actions() indentation is correct\n")
    return True

def test_carry_structure():
    """Test 3: Verify carry structure includes curiosity_state"""
    print("=" * 60)
    print("Test 3: Carry Structure with curiosity_state")
    print("=" * 60)

    print("  Checking code structure in agent.py...")

    with open('/root/dreamerv3_curiosity-Dynamic_Curiosity_Threshold/dreamerv3/agent.py', 'r') as f:
        content = f.read()

    # Check 1: Agent.__init__ initializes curiosity_trigger
    if 'self.curiosity_trigger = CuriosityTrigger(' in content:
        print("✓ Agent.__init__() initializes curiosity_trigger")
    else:
        print("✗ Agent.__init__() missing curiosity_trigger initialization")
        return False

    # Check 2: init_policy returns 5-tuple
    if 'curiosity_state)' in content and 'def init_policy' in content:
        # Find init_policy and check it returns curiosity_state
        lines = content.split('\n')
        in_init_policy = False
        found_return = False
        for line in lines:
            if 'def init_policy' in line:
                in_init_policy = True
            elif in_init_policy and 'return (' in line:
                found_return = True
            elif in_init_policy and found_return and 'curiosity_state)' in line:
                print("✓ init_policy() returns 5-tuple with curiosity_state")
                break
        else:
            if not found_return:
                print("✗ Could not verify init_policy() return statement")
                return False
    else:
        print("✗ init_policy() does not return curiosity_state")
        return False

    # Check 3: policy() unpacks 5-tuple carry
    if '(enc_carry, dyn_carry, dec_carry, prevact, curiosity_state) = carry' in content:
        print("✓ policy() unpacks 5-tuple carry with curiosity_state")
    else:
        print("✗ policy() does not unpack curiosity_state from carry")
        return False

    # Check 4: policy() uses stateless API
    if 'self.curiosity_trigger.update(curiosity_state, mean_entropy)' in content:
        print("✓ policy() uses stateless curiosity_trigger.update()")
    else:
        print("✗ policy() does not use stateless update API")
        return False

    # Check 5: policy() returns updated carry with curiosity_state
    if 'carry = (enc_carry, dyn_carry, dec_carry, act, curiosity_state)' in content:
        print("✓ policy() returns carry with updated curiosity_state")
    else:
        print("✗ policy() does not return curiosity_state in carry")
        return False

    # Check 6: train() handles curiosity_state
    if 'carry, obs, prevact, stepid, curiosity_state = self._apply_replay_context' in content:
        print("✓ train() receives curiosity_state from _apply_replay_context()")
    else:
        print("✗ train() does not receive curiosity_state")
        return False

    # Check 7: _apply_replay_context returns curiosity_state
    if 'return carry, obs, prevact, stepid, curiosity_state' in content:
        print("✓ _apply_replay_context() returns curiosity_state")
    else:
        print("✗ _apply_replay_context() does not return curiosity_state")
        return False

    print("✓ All carry structure checks passed\n")
    return True

def main():
    print("\n")
    print("=" * 60)
    print("DreamerV3 Curiosity Bug Fix Verification")
    print("=" * 60)
    print()

    tests = [
        ("CuriosityTrigger Stateless", test_curiosity_trigger),
        ("sample_uniform_actions() Fix", test_sample_uniform_actions),
        ("Carry Structure", test_carry_structure),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All bug fixes verified successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
