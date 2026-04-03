"""
End-to-End Demo for Agentic Knowledge Distillation (Knowledge Pyramid).
This script builds the environment, ingests mock financial data, builds the pyramid,
and runs demonstrative intelligent queries.
"""

from embedder import Embedder
from pyramid import KnowledgePyramidBuilder
from retrieval import PyramidRetriever

# 1. Mock Data Source (Financial Report for OmniTech Corp)
MOCK_PAGES = [
    # Page 1 — Pure Finance
    """OmniTech Corp Q3 2024 Financial Report.
    OmniTech reported record third-quarter revenue of $24.5 billion, 
    an 18% year-over-year increase. Net income was $7.2 billion. 
    Operating margins expanded to 34%. Earnings per share rose to $3.10.
    The board approved a quarterly dividend of $0.85 per share.
    Free cash flow reached $6.1 billion for the quarter.""",

    # Page 2 — Pure AI Strategy  
    """AI Strategy and Market Positioning.
    OmniTech's strategy positions the company as the foundational AI layer 
    for enterprise adoption. Our AI strategy accelerated capital expenditures 
    to $4.1 billion dedicated entirely to AI compute infrastructure.
    By integrating generative AI natively into our software suite, our 
    AI strategy provides a seamless upgrade path for 50,000 enterprise customers.
    The company's AI positioning significantly lowers barriers compared to 
    standalone AI solutions. AI investments will continue through 2025.""",

    # Page 3 — Pure Risk & Compliance
    """Risk Factors and Legal Compliance Obligations.
    Management identified several critical regulatory and compliance risks.
    Legal compliance with the EU AI Act requires structural changes.
    Regulatory risk from data privacy laws across multiple jurisdictions 
    creates significant legal liability exposure.
    Compliance costs are projected to increase 40% over 24 months.
    The audit committee reviews all compliance obligations quarterly.
    Legal counsel has been retained for all regulatory risk matters.""",
]

def print_ingestion_report(nodes):
    print("\n" + "="*50)
    print("INGESTION QUALITY REPORT")
    print("="*50)
    
    chunks = set(n["chunk_id"] for n in nodes)
    print(f"Pages processed:      {len(MOCK_PAGES)}")
    print(f"Chunks created:       {len(chunks)} (2-page sliding window)")
    print(f"Total Pyramid nodes:  {len(nodes)}")
    print("-" * 50)
    
    categories = {}
    for n in nodes:
        if n["layer"] == "Layer 3: Category":
            cat = n.get("category_name", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
    print("Category Distribution (Chunk Topics):")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} chunk(s)")
        
    print("="*50 + "\n")

def run_demo():
    print("Initializing Knowledge Pyramid System...\n")
    
    # Initialize components
    embedder = Embedder('all-MiniLM-L6-v2')
    builder = KnowledgePyramidBuilder(embedder)
    
    # Build Pyramid
    pyramid_nodes = builder.build_pyramid(MOCK_PAGES)
    
    # Print what the system "learned" about the dataset globally
    print_ingestion_report(pyramid_nodes)
    
    retriever = PyramidRetriever(embedder, pyramid_nodes)
    
    # Test Queries designed to test different layers
    test_queries = [
        # Fact-finding -> Should hit Layer 4 (Keywords) or Layer 1
        "What was OmniTech's revenue in Q3?",
        
        # Thematic/Risk -> Should hit Layer 3 (Category)
        "What compliance and legal risks does the company face?",
        
        # Analytical -> Should hit Layer 2 (Summary)
        "How is the company positioning its AI strategy?"
    ]
    
    for q in test_queries:
        print(f"\nQUERY: '{q}'")
        intent = retriever.detect_query_intent(q)
        print(f"Detected Intent / Boost: {intent if intent else 'General search'}")
        
        results = retriever.search(q, top_k=1, fuse_scores=True)
        
        if results:
            best = results[0]
            node = best["node"]
            confidence = "HIGH" if best['score'] > 0.6 else ("MEDIUM" if best['score'] > 0.4 else "LOW")
            
            print(f"Verdict:     {confidence} CONFIDENCE MATCH (Score: {best['score']:.2f})")
            print(f"Matched at:  {node['layer']}")
            print(f"Source Pages: {node['pages']}")
            if best.get('fusion_applied'):
                print(f"Note:        Cross-layer reciprocal rank fusion applied")
            print(f"Node Content: {node['text']}")
            
            # Show what is actually sent to the LLM
            context = node.get('raw_text', '').replace('\n', ' ')
            if len(context) > 120: context = context[:117] + '...'
            print(f"RAG Context: {context}")
        else:
            print("No results found.")
            
    print("\nDemo completed.")

if __name__ == "__main__":
    run_demo()
