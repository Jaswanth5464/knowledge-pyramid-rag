> Traditional RAG retrieves chunks. This system retrieves **understanding**.

# Agentic Knowledge Pyramid RAG System 

A production-ready pipeline for "Agentic Knowledge Distillation + Pyramid Search" that inverts the traditional RAG tradeoff: instead of spending maximum compute at query-time trying to analyze raw document chunks, we spend compute once at ingestion-time to distill the dataset into a multi-layered pyramid of cognitive understanding.

## The Problem With Traditional RAG
Most traditional RAG pipelines ingest text blindly: they take a document, chunk it into 500-token blocks, embed them, and index them. When a query comes in, they perform unstructured similarity search over raw prose. 
This works for simple queries but fails dramatically on:
1. **Thematic Questions**: ("What is their AI strategy?")
2. **Fact-Finding across Noise**: ("What was Q3 revenue?")

## Architecture Diagram

```mermaid
graph TD
    classDef input fill:#2c3e50,stroke:#34495e,color:#fff,stroke-width:2px,rx:5px,ry:5px;
    classDef process fill:#34495e,stroke:#2c3e50,color:#fff,stroke-width:2px,rx:5px,ry:5px;
    classDef knowledge fill:#27ae60,stroke:#2ecc71,color:#fff,stroke-width:2px,rx:5px,ry:5px;
    classDef search fill:#8e44ad,stroke:#9b59b6,color:#fff,stroke-width:2px,rx:5px,ry:5px;
    classDef intent fill:#c0392b,stroke:#e74c3c,color:#fff,stroke-width:2px,rx:5px,ry:5px;

    %% Ingestion Phase
    Docs[Document Pages]:::input -->|2-Page Sliding Window| Chunker[Chunker]:::process
    
    subgraph Knowledge Pyramid
        Chunker --> L1[Layer 1: Raw Text]:::knowledge
        Chunker --> L2[Layer 2: Chunk Summary]:::knowledge
        Chunker --> L3[Layer 3: Category / Theme]:::knowledge
        Chunker --> L4[Layer 4: Distilled Keywords]:::knowledge
    end
    
    %% Storage
    L1 -.-> DB[(Vector Database)]
    L2 -.-> DB
    L3 -.-> DB
    L4 -.-> DB
    
    %% Retrieval Phase
    Query[User Query]:::input --> Intent{Intent Classifier}:::intent
    Query --> Embed[Embed Query]:::process
    
    Embed --> Search[Cosine Sim Search]:::search
    DB --> Search
    Intent -->|Boosts target layer| Search
    
    Search --> RRF[RRF Score Fusion]:::process
    RRF -->|Cross-layer merge| Results[Ranked Results]:::input
```

## How It Works (A Simple Example)

If you hand a traditional AI a 500-page corporate document and ask, *"What was Q3 revenue?"*, it blindly chops the document into 1,000 random paragraphs. It then uses basic math to guess which paragraph looks the most like your question. It frequently fails because the word "revenue" might be in one paragraph, but the actual dollar amount is in the next. The context is destroyed.

**Our system acts like a professional librarian. It processes the document *before* anyone is allowed to ask a question:**

1. **Smart Ingestion (The Sliding Window):** 
   Instead of chopping pages randomly, the system reads Page 1 and Page 2 together. Then it reads Page 2 and Page 3 together. This overlapping "sliding window" guarantees that a sentence split across two pages is never taken out of context.

2. **Building the Pyramid (Distillation):** 
   While reading those pages, it doesn't just save the raw text. It builds a meticulously organized 4-layer "Knowledge Pyramid" index for that specific chunk:
   - **Layer 1 (Raw Text):** The exact, full-fidelity words on the page.
   - **Layer 2 (Summary):** It automatically generates a compressed 2-sentence summary of the page.
   - **Layer 3 (Category):** It flags the page with a thematic sticky note (like *[Risk & Compliance]*).
   - **Layer 4 (Keywords):** It extracts a list of the most mathematically significant, factual words.

3. **Intent Routing (The Detective):** 
   When a user finally asks *"What was Q3 revenue?"*, the system analyzes the question first. It realizes this is a "fact-finding" inquiry. Instead of searching all the broad text, it proactively tells the search engine: *"Ignore the thematic summaries; prioritize searching the Layer 4 Keywords."*

4. **Score Fusion (Double-Checking the Evidence):** 
   The semantic search engine evaluates the text. If it finds strong evidence for the answer in the Layer 4 Keywords, AND it also finds supporting evidence in the Layer 2 Summary of the exact same chunk, it statistically fuses those scores together using **Reciprocal Rank Fusion (RRF)**. 

**The Result:** The system retrieves a final answer with a mathematically sound "High Confidence" score. It avoids hallucinations entirely because it mapped the cognitive intent of the user's question directly to the correct abstraction layer of the data.

## The Solution: A Pre-Computed Reasoning Cache
This system creates an **Agentic Knowledge Pyramid**. By using a 2-page sliding window (to solve edge-context loss), every chunk of text is analyzed and distilled into 4 distinct "zoom levels":

- **Layer 1: Raw Text** (Full fidelity context)
- **Layer 2: Summary** (Context-aware compression for broad reasoning)
- **Layer 3: Category / Theme** (Rule-based categorization for thematic routing)
- **Layer 4: Distilled Keywords** (High-signal atomic facts with noise removed)

## Project Structure

```text
knowledge_pyramid/
│
├── README.md          ← you are here
├── pyramid.py         ← sliding window + 4-layer builder
├── embedder.py        ← sentence-transformers + cosine similarity  
├── retrieval.py       ← intent routing + RRF fusion
├── demo.py            ← sample document + test queries
└── requirements.txt   ← dependencies
```

## Why This Implementation is Unique

**1. Cross-Layer Reciprocal Rank Fusion (RRF)**
If Chunk B scores `0.7` on its Summary, and `0.85` on its Keywords, our `retrieval.py` mathematically fuses these signals. It recognizes that "supporting evidence across multiple abstractions" indicates a definitive match, yielding a highly confident system.

**2. Intent-Based Query Routing**
Not all queries are created equal. A short factual query ("Q3 Revenue") shouldn't be matched against a generic summary, and a broad thematic query ("Risk factors") shouldn't be matched against raw text. Our system uses a lightweight heuristic classifier to detect the *shape* of the query and boosts the target abstraction layer dynamically.

**3. Semantic Embeddings over Fuzzy Matching**
Fuzzy match focuses on overlapping characters. We utilized the `sentence-transformers` library (`all-MiniLM-L6-v2`) to capture true **semantic meaning**. This means "earnings" and "revenue" map to the same vector logic, enabling human-like retrieval without the fragility of string matching.

**4. Ingestion Quality Report**
The system builds a meta-understanding of its own dataset before ever evaluating a query, allowing the pipeline to self-diagnose what it "learned."

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
# Alternatively: pip install sentence-transformers numpy scikit-learn
```

## Usage & Execution Results

Run the demonstration script to see the pyramid in action:
```bash
python demo.py
```

### Ingestion Quality Report
*The system understands its own knowledge boundaries before answering a single query.*



<img width="1083" height="535" alt="image" src="https://github.com/user-attachments/assets/795cbdc0-f8fc-420f-83bf-139b5145d6d7" />




### Query Execution Results
*Reciprocal Rank Fusion combines evidence across abstraction levels, while Intent-based routing ensures each query searches at the right cognitive zoom level.*



<img width="974" height="515" alt="image" src="https://github.com/user-attachments/assets/418e9c40-a41d-4512-9d80-b7e2f7708b9a" />




## Limitations & Future Work
Every robust architecture has constraints. Here is where this implementation breaks down and how to scale it:
* **Thematic Rigidity**: The Category layer currently uses hardcoded keyword rules. This fails if a chunk discusses Finance without using explicitly listed finance-words. *Fix: Use a zero-shot LLM classifier.*
* **Sliding Window Redundancy**: A fixed step size of 1 creates overlapping redundant nodes. For massive documents, this becomes increasingly expensive at search-time. *Fix: Implement dynamic chunk boundaries using semantic-similarity calculation drops between paragraphs.*

## Core Insight

The pyramid doesn't just store documents differently.  
It stores them at **multiple levels of human-like understanding** — so the retrieval system doesn't search text, it searches cognition.

This is the foundation for enterprise RAG systems that actually work on complex analytical questions, effectively shifting the cognitive burden from query-time hallucination mitigation to ingestion-time knowledge distillation!

---

## Output Profile & Contact
**Jaswanth Kanamrlapudi**
* 📧 Email: [jaswanth5464@gmail.com](mailto:jaswanth5464@gmail.com)
* 💼 LinkedIn: [jaswanth-kanamrlapudi-a41197252](https://www.linkedin.com/in/jaswanth-kanamrlapudi-a41197252)
* 💻 GitHub: [@Jaswanth5464](https://github.com/Jaswanth5464)
