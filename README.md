# Deal Intelligence for Sales Teams

A decision intelligence system for sales teams — combining predictive ML, temporal reasoning, and multi-agent debate to diagnose deal risk and optimize sales strategy.

## Overview

This is a Revenue Intelligence Copilot: a system that scores deals, explains why they fail, recommends actions, and improves over time. It combines:

- **Predictive ML** — XGBoost-based win probability scoring with calibrated outputs
- **Temporal Feature Engineering** — time-series signals like engagement decay, silence gaps, and deal velocity
- **Multi-Agent Reasoning** — collaborative agents that debate deal strategy using LangGraph
- **RAG-Grounded Analysis** — risk assessments backed by historical deal evidence, not hallucinations
- **Evaluation Harness** — documented model performance, A/B tested agent impact, and named failure modes

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                         │
│  (Pipeline View | Deal Detail | Agent Debate | Simulator)│
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend                        │
│  /analyze-deal | /predict-outcome | /generate-strategy   │
└──────────┬───────────────┬───────────────┬──────────────┘
           │               │               │
┌──────────▼───┐  ┌───────▼───────┐  ┌───▼──────────────┐
│  ML Layer    │  │  Agent Layer  │  │  RAG Layer       │
│  (XGBoost)   │  │  (LangGraph)  │  │  (FAISS/Pinecone)│
└──────────────┘  └───────────────┘  └──────────────────┘
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python, FastAPI |
| ML | XGBoost, scikit-learn, SHAP |
| Agent Framework | LangGraph |
| LLM (Hybrid) | OpenAI API + Llama 3.2 (local) |
| Retrieval | FAISS / Pinecone, OpenAI embeddings |
| Database | PostgreSQL |
| Frontend | React, Tailwind CSS, Recharts |
| Evaluation | Custom harness, Weights & Biases |

## Project Structure

```
├── backend/
│   ├── api/              # FastAPI routes
│   ├── agents/           # LangGraph multi-agent system
│   ├── ml/               # ML models and training
│   ├── features/         # Temporal feature engineering
│   ├── rag/              # Vector store and retrieval
│   └── evaluation/       # Evaluation harness
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page views
│   │   └── utils/        # Utilities
│   └── public/
├── data/
│   ├── raw/              # Raw CRM data (synthetic)
│   └── processed/        # Engineered features
├── notebooks/            # Exploration and analysis
├── tests/                # Test suite
└── docs/                 # Documentation
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (optional, SQLite for development)

### Installation

```bash
# Clone the repository
git clone https://github.com/Adeliyio/deal-intelligence-for-sales-teams.git
cd deal-intelligence-for-sales-teams

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### Running

```bash
# Start backend
cd backend
uvicorn api.main:app --reload

# Start frontend (separate terminal)
cd frontend
npm run dev
```

## Key Design Decisions

- **XGBoost over neural networks**: CRM data is tabular and sparse. Tree-based models handle missing values naturally and outperform MLPs at realistic dataset sizes.
- **Binary risk classification over survival analysis**: We model whether a deal stalls, not when it closes.
- **Critic Agent with measured impact**: The Critic's value is A/B tested, not assumed.
- **Cold-start aware**: Designed for 50-100 deals with explicit confidence intervals.
- **Hybrid LLM routing**: OpenAI for reasoning-critical tasks, Llama 3.2 for lightweight processing.

## License

MIT
