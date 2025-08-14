import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------
# PSI Function
# ---------------------------
def calculate_psi(expected, actual, buckets=10, eps=1e-4):
    """Calculate Population Stability Index between two samples."""
    expected = np.array(expected)
    actual = np.array(actual)

    # Remove NaNs
    expected = expected[~pd.isnull(expected)]
    actual = actual[~pd.isnull(actual)]

    # If non-numeric, handle as categorical
    try:
        expected = expected.astype(float)
        actual = actual.astype(float)
    except ValueError:
        # Categorical PSI
        exp_perc = pd.Series(expected).value_counts(normalize=True)
        act_perc = pd.Series(actual).value_counts(normalize=True)
        exp_perc, act_perc = exp_perc.align(act_perc, fill_value=0)
        exp_perc = exp_perc.replace(0, eps)
        act_perc = act_perc.replace(0, eps)
        return ((exp_perc - act_perc) * np.log(exp_perc / act_perc)).sum()

    # Create bins based on quantiles of expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    # Assign bins
    expected_bins = pd.cut(expected, bins=np.concatenate(([-np.inf], breakpoints[1:-1], [np.inf])))
    actual_bins = pd.cut(actual, bins=np.concatenate(([-np.inf], breakpoints[1:-1], [np.inf])))

    # Convert to Series before value_counts (fix for pandas versions)
    exp_perc = pd.Series(expected_bins).value_counts(normalize=True)
    act_perc = pd.Series(actual_bins).value_counts(normalize=True)

    # Align bins
    exp_perc, act_perc = exp_perc.align(act_perc, fill_value=0)

    # Avoid log(0)
    exp_perc = exp_perc.replace(0, eps)
    act_perc = act_perc.replace(0, eps)

    # PSI formula
    psi_value = ((exp_perc - act_perc) * np.log(exp_perc / act_perc)).sum()
    return psi_value

# ---------------------------
# Load dataset (local path)
# ---------------------------
df = pd.read_csv("data.csv")  # Ensure this file is in the same folder as your notebook

# Target column
target_col = "Bankrupt?"

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target_col])

# ---------------------------
# Calculate PSI for each feature
# ---------------------------
psi_results = {}
for col in df.columns:
    if col == target_col:
        continue
    psi_results[col] = calculate_psi(train_df[col], test_df[col])

# ---------------------------
# Show results
# ---------------------------
psi_df = pd.DataFrame(list(psi_results.items()), columns=["Feature", "PSI"])
psi_df["Drift Interpretation"] = pd.cut(
    psi_df["PSI"],
    bins=[-np.inf, 0.1, 0.25, np.inf],
    labels=["No drift", "Moderate drift", "Significant drift"]
)

print("\nPSI Results (Train vs Test):")
print(psi_df.sort_values(by="PSI", ascending=False))
