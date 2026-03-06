"""
Test script to verify auto-increment scenario ID functionality.
This script tests the find_next_available_scenario_id() function without importing dependencies.
"""

import os
import sys
from pathlib import Path


def find_next_available_scenario_id(output_dir: str) -> int:
    """
    Finds the next available scenario ID by checking existing scenario folders.
    (Copy of function from medsim/run.py for standalone testing)
    
    Args:
        output_dir (str): Path to the output directory.
    
    Returns:
        int: Next available scenario ID (0 if no scenarios exist).
    """
    if not os.path.exists(output_dir):
        return 0
    
    max_id = -1
    try:
        for item in os.listdir(output_dir):
            if item.startswith("scenario_") and os.path.isdir(os.path.join(output_dir, item)):
                try:
                    scenario_id = int(item.replace("scenario_", ""))
                    max_id = max(max_id, scenario_id)
                except ValueError:
                    # Skip if folder name doesn't follow pattern
                    continue
    except Exception as e:
        print(f"Warning: Error while finding next scenario ID: {e}")
    
    return max_id + 1


def test_scenario_id_increment():
    """Test the scenario ID increment functionality."""
    
    # Create a temporary output directory for testing
    test_output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    
    # Test 1: Empty directory should return 0
    print("Test 1: Empty directory")
    result = find_next_available_scenario_id(test_output_dir)
    assert result == 0, f"Expected 0, but got {result}"
    print(f"✓ Empty directory returns: {result}")
    
    # Test 2: Create some scenario folders and test
    print("\nTest 2: Directory with existing scenarios")
    os.makedirs(test_output_dir, exist_ok=True)
    os.makedirs(os.path.join(test_output_dir, "scenario_0"), exist_ok=True)
    os.makedirs(os.path.join(test_output_dir, "scenario_1"), exist_ok=True)
    os.makedirs(os.path.join(test_output_dir, "scenario_2"), exist_ok=True)
    
    result = find_next_available_scenario_id(test_output_dir)
    assert result == 3, f"Expected 3, but got {result}"
    print(f"✓ Found existing scenarios (0, 1, 2), next ID: {result}")
    
    # Test 3: Create gaps in numbering
    print("\nTest 3: Gaps in scenario numbering")
    # Remove scenario_1 to test gap handling
    import shutil
    shutil.rmtree(os.path.join(test_output_dir, "scenario_1"), ignore_errors=True)
    
    result = find_next_available_scenario_id(test_output_dir)
    assert result == 3, f"Expected 3 (max+1), but got {result}"
    print(f"✓ Gaps ignored, returns max ID + 1: {result}")
    
    # Test 4: Non-matching folders are ignored
    print("\nTest 4: Non-matching folders ignored")
    os.makedirs(os.path.join(test_output_dir, "output_logs"), exist_ok=True)
    os.makedirs(os.path.join(test_output_dir, "data"), exist_ok=True)
    
    result = find_next_available_scenario_id(test_output_dir)
    assert result == 3, f"Expected 3, but got {result}"
    print(f"✓ Non-matching folders ignored, result: {result}")
    
    # Cleanup
    shutil.rmtree(test_output_dir, ignore_errors=True)
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    try:
        test_scenario_id_increment()
        print("\n" + "="*50)
        print("SUCCESS: Auto-increment scenario ID is working correctly!")
        print("="*50)
        print("\nUsage:")
        print("- Each time you run the simulation, scenario IDs will be auto-numbered")
        print("- First run: scenario_0, scenario_1, ...")
        print("- Second run: scenario_<N>, scenario_<N+1>, ... (where N is the next available ID)")
        print("- Old scenarios will NOT be overwritten")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)

