import numpy as np

# Denavit-Hartenberg Transformation Matrix Function
def dh_matrix(theta, d, a, alpha):
    th = np.radians(theta)
    al = np.radians(alpha)
    return np.array([
        [np.cos(th), -np.sin(th)*np.cos(al),  np.sin(th)*np.sin(al), a*np.cos(th)],
        [np.sin(th),  np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
        [0,           np.sin(al),             np.cos(al),            d],
        [0,           0,                      0,                     1]
    ])

# Helper function to run and print tests cleanly
def run_validation_test(test_name, dh_params):
    print(f"=== {test_name} ===")
    T_final = np.eye(4)
    
    for i, p in enumerate(dh_params):
        T_i = dh_matrix(p[0], p[1], p[2], p[3])
        T_i[np.abs(T_i) < 1e-10] = 0  # Clean up near-zero floating point values
        
        print(f"T_{i}{i+1} Matrix:")
        print(np.round(T_i, 4))
        print("-" * 30)
        
        T_final = np.dot(T_final, T_i)

    T_final[np.abs(T_final) < 1e-10] = 0
    print("T_06 (Final) Matrix:")
    print(np.round(T_final, 4))
    print("\n" + "="*40 + "\n")


# ---------------------------------------------------------
# CROSS-VALIDATION TEST SCENARIOS
# ---------------------------------------------------------

# Test 1: Home Position (All angles 0)
dh_params_test1 = [
    [0.0, 0.400, 0.025, -90],
    [0.0, 0.000, 0.455,   0],
    [0.0, 0.000, 0.035, -90],
    [0.0, 0.420, 0.000,  90],
    [0.0, 0.000, 0.000, -90],
    [0.0, 0.080, 0.000,   0]
]
run_validation_test("Test 1: Home Position", dh_params_test1)


# Test 2: Arbitrary Configuration
dh_params_test2 = [
    [45.0,  0.400, 0.025, -90],
    [-30.0, 0.000, 0.455,   0],
    [60.0,  0.000, 0.035, -90],
    [90.0,  0.420, 0.000,  90],
    [0.0,   0.000, 0.000, -90],
    [-45.0, 0.080, 0.000,   0]
]
run_validation_test("Test 2: Arbitrary Configuration", dh_params_test2)


# Test 3: Asymmetric Wrist
dh_params_test3 = [
    [0.0,   0.400, 0.025, -90],
    [0.0,   0.000, 0.455,   0],
    [0.0,   0.000, 0.035, -90],
    [45.0,  0.420, 0.000,  90],
    [-60.0, 0.000, 0.000, -90],
    [30.0,  0.080, 0.000,   0]
]
run_validation_test("Test 3: Asymmetric Wrist", dh_params_test3)


# Test 4: Base Isolation
dh_params_test4 = [
    [90.0, 0.400, 0.025, -90],
    [0.0,  0.000, 0.455,   0],
    [0.0,  0.000, 0.035, -90],
    [0.0,  0.420, 0.000,  90],
    [0.0,  0.000, 0.000, -90],
    [0.0,  0.080, 0.000,   0]
]
run_validation_test("Test 4: Base Isolation", dh_params_test4)
