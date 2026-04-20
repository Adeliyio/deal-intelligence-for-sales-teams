"""
Build the FAISS vector store from processed deal features.

Run from project root: python -m scripts.build_vector_store
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backend.rag.vector_store import DealVectorStore
from backend.rag.retriever import DealRetriever


def main():
    print("=" * 60)
    print("Deal Intelligence — Vector Store Builder")
    print("=" * 60)

    # Load features
    features_path = "data/processed/deal_features.csv"
    print(f"\nLoading features from {features_path}...")
    df = pd.read_csv(features_path)

    # Convert booleans
    if "has_economic_buyer" in df.columns:
        df["has_economic_buyer"] = df["has_economic_buyer"].astype(int)

    print(f"Loaded {len(df)} deals")

    # Build index
    print("\nBuilding FAISS index (feature-based embeddings)...")
    store = DealVectorStore(embedding_mode="feature")
    store.build_index(df)
    print(f"Index built: {store.index.ntotal} vectors, dimension {store.index.d}")

    # Save index
    store.save()
    print(f"Index saved to data/vector_store/faiss_index/")

    # Demo: retrieval
    print("\n" + "=" * 60)
    print("Retrieval Demo")
    print("=" * 60)

    retriever = DealRetriever(store)

    # Pick a sample deal
    sample_deal = df.iloc[0].to_dict()
    print(f"\nQuery deal: {sample_deal['deal_id']} (outcome: {sample_deal['outcome']})")
    print(f"  Industry: {sample_deal.get('industry')}, Size: {sample_deal.get('deal_size_bucket')}")
    print(f"  Engagement/week: {sample_deal.get('engagement_per_week', 0):.3f}")
    print(f"  Silence gap: {sample_deal.get('silence_gap_days', 0):.0f} days")

    # Similar won deals
    print("\n--- Similar Won Deals ---")
    won = retriever.store.find_similar_won_deals(sample_deal, top_k=3)
    if won:
        for d in won:
            print(f"  {d['deal_id']} | similarity: {d['similarity']:.3f} | "
                  f"engagement: {d.get('engagement_per_week', 0):.3f}/wk | "
                  f"duration: {d.get('duration_days', 0)}d")
    else:
        print("  No similar won deals found above threshold")

    # Similar lost deals
    print("\n--- Similar Lost Deals ---")
    lost = retriever.store.find_similar_lost_deals(sample_deal, top_k=3)
    if lost:
        for d in lost:
            print(f"  {d['deal_id']} | similarity: {d['similarity']:.3f} | "
                  f"silence: {d.get('silence_gap_days', 0):.0f}d | "
                  f"duration: {d.get('duration_days', 0)}d")
    else:
        print("  No similar lost deals found above threshold")

    # Cohort insights
    print("\n--- Cohort Insights ---")
    insights = retriever.store.get_cohort_insights(
        industry=sample_deal.get("industry", ""),
        deal_size_bucket=sample_deal.get("deal_size_bucket", ""),
    )
    for key, val in insights.items():
        print(f"  {key}: {val}")

    # Graceful degradation check
    print("\n--- RAG Confidence Check ---")
    check = retriever.graceful_degradation_check(sample_deal)
    for key, val in check.items():
        print(f"  {key}: {val}")

    print("\n\nVector store build complete.")


if __name__ == "__main__":
    main()
