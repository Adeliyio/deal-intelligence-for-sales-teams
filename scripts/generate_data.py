"""
Data generation pipeline.

Generates synthetic CRM data and computes temporal features.
Run from the project root: python -m scripts.generate_data
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.features.synthetic_data_generator import SyntheticCRMGenerator
from backend.features.temporal_features import TemporalFeatureEngineer


def main():
    print("=" * 60)
    print("Deal Intelligence — Data Generation Pipeline")
    print("=" * 60)

    # Step 1: Generate synthetic CRM data
    print("\n[1/3] Generating synthetic CRM data...")
    generator = SyntheticCRMGenerator(n_deals=75, seed=42)
    data = generator.generate_all()
    generator.save_to_csv(output_dir="data/raw")

    # Step 2: Compute temporal features
    print("\n[2/3] Computing temporal features...")
    engineer = TemporalFeatureEngineer(
        deals_df=data["deals"],
        activities_df=data["activities"],
        contacts_df=data["contacts"],
        stage_transitions_df=data["stage_transitions"],
    )
    features_df = engineer.compute_all_features()

    # Step 3: Save feature matrix
    output_path = "data/processed/deal_features.csv"
    os.makedirs("data/processed", exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"\n[3/3] Saved feature matrix: {output_path}")
    print(f"  - Shape: {features_df.shape}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Feature Summary")
    print("=" * 60)

    numeric_cols = features_df.select_dtypes(include="number").columns
    summary = features_df[numeric_cols].describe().round(3)
    print(summary.to_string())

    # Outcome distribution
    print("\n\nOutcome Distribution:")
    print(features_df["outcome"].value_counts().to_string())

    # Null counts
    null_counts = features_df.isnull().sum()
    if null_counts.any():
        print("\n\nNull Counts (features with missing values):")
        print(null_counts[null_counts > 0].to_string())

    print("\n\nPipeline complete.")


if __name__ == "__main__":
    main()
