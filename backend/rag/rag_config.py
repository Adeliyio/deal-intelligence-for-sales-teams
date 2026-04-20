"""
RAG layer configuration — vector store settings, embedding parameters,
and retrieval tuning.
"""

# Vector store backend
VECTOR_STORE_TYPE = "faiss"  # "faiss" (local) or "pinecone" (production)

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# FAISS index settings
FAISS_INDEX_PATH = "data/vector_store/faiss_index"
FAISS_METRIC = "cosine"  # cosine similarity for semantic search

# Retrieval settings
TOP_K_RESULTS = 5  # Number of similar deals to retrieve
SIMILARITY_THRESHOLD = 0.65  # Minimum similarity to include in results
MAX_CONTEXT_TOKENS = 2000  # Max tokens for retrieved context

# Document chunking (for activity logs and notes)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Deal record fields to embed
DEAL_EMBED_FIELDS = [
    "deal_id",
    "outcome",
    "industry",
    "deal_size_bucket",
    "deal_value",
    "duration_days",
    "stage",
    "source",
]

# Feature fields to include in deal embeddings (for similarity matching)
FEATURE_EMBED_FIELDS = [
    "engagement_score",
    "engagement_per_week",
    "avg_response_time_hours",
    "silence_gap_days",
    "silence_gap_severity",
    "stakeholder_count",
    "has_economic_buyer",
    "decision_maker_ratio",
    "deal_velocity_ratio",
    "decay_weighted_engagement",
]

# Query templates for structured retrieval
QUERY_TEMPLATES = {
    "similar_outcome": "Find deals similar to {deal_id} that ended with outcome {outcome}",
    "silence_pattern": "Find deals with silence gap pattern similar to {silence_gap_days} days at {deal_age} weeks",
    "successful_strategy": "What did successful reps do differently with {industry} {deal_size_bucket} deals?",
    "risk_comparison": "Find deals that looked like this at day {duration_days} and ultimately churned",
}
