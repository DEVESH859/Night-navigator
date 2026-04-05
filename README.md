# Fear-Free Night Navigator 🛡️🌙

**Fear-Free Night Navigator** is an AI-powered safety application that addresses the blind spot in standard navigation tools (which blindly optimize for ETA) by introducing **Psychological Safety** as the primary routing metric. 

Designed for solo travelers and night commuters, Night Navigator quantitatively evaluates street-level metrics (lighting, commercial footfall, road characteristics, police presence, and historical incident risk) to build a multi-layered safety score, finding routes that optimize for peace of mind over speed.

---

## 🚀 Hackathon Submission Context

This prototype was developed to meet the highly technical engineering standards of the Google Hackathon, shifting beyond conceptual wrappers into an architecturally scalable and algorithmically sound machine learning deployment.

### 1. Core Logic & AI Architecture
* **Algorithmic Routing:** We utilize graph theory (`networkx` and `osmnx`) to construct a multi-directional graph of the city's road network. The engine employs **A* (A-Star) search** equipped with a custom composite cost function `(alpha * distance + beta * safety_penalty)` to algorithmically balance navigation distance with safety. 
* **Model Approach:** Our underlying ML pipeline uses **Random Forest** and **XGBoost classifiers/regressors** to identify complex non-linear structures in our generated proxy data. 
* **Agentic Execution:** User natural-language queries are parsed via an intelligent **LangGraph / LangChain supervisor LLM** which orchestrates intent, searches for real-time crime incidents (Agent 2), explains safety metrics (Agent 3), and aggregates an enriched response payload.

### 2. Demonstrable Reliability & Evaluation
* **Quantitative Routing Evaluation:** Included is an explicit evaluation suite (`evaluate.py`) that tests 20 random Origin-Destination coordinates to calculate explicit trade-offs. (e.g. *Distance Overhead % vs. Safety Gain %*).
* **Explainable AI (XAI):** The system relies on **SHAP (SHapley Additive exPlanations)** values and ablation studies to explicitly demonstrate feature importance and validate the integrity of our model against the black-box problem. Residual distribution plots exist to verify regression health.

### 3. Quality over Polish
Our development timeline overwhelmingly prioritized the mathematics and reliability of the platform's backend infrastructure. Rather than investing deeply in a front-end wrapper, we focused on robust API endpoints (via **FastAPI**), LLM intent fallback chains, exception handling on A* routing failure modes, and automated metric benchmarking for our ML pipelines.

### 4. Data Strategy
Because high-quality granular street-level crime and psychological safety maps are not publicly accessible or cleanly digitized, we executed a rigorous proxy and synthetic data approach. We synthesized variables denoting **commercial footfall**, **street lighting**, and **police proximity**, augmenting real graph anchor nodes with 5,000 statistically distributed synthetic rows mirroring target variables. This enabled our models to functionally isolate which topological configurations best minimize transit anxiety.

---

## 🛠️ Tech Stack
* **Core API & Logic:** Python, FastAPI, NetworkX, OSMnx
* **AI/ML:** Scikit-Learn (Random Forest), XGBoost, SHAP, LangChain, LangGraph
* **LLM Engine:** Groq (Llama-3.1 8B), Mistral (OpenRouter failovers)

## 📌 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DEVESH859/night-navigator.git
   cd night-navigator
   ```

2. **Install dependancies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Create a `.env` file at the root of the directory and include your API Keys:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   TAVILY_API_KEY="your_tavily_api_key_here"
   OPENROUTER_API_KEY="your_openrouter_api_key_here"
   ```

4. **Launch the Core Backend Services:**
   ```bash
   python -m uvicorn api.main:app --port 8000 --reload
   ```

5. **Generate & Compile Machine Learning Data (Optional):**
   ```bash
   python models/train_safety_model.py
   python evaluation/evaluate.py
   ```
