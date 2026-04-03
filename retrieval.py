"""
Smart retrieval mechanism that queries the Knowledge Pyramid.
Instead of treating all nodes equally, it uses heuristics to route queries to the 
most appropriate layer (e.g. "what was revenue" -> Layer 4 Keywords), and fuses
scores across layers if a single chunk scores highly in multiple abstractions
(Reciprocal Rank Fusion concept).
"""

import collections
import numpy as np
from typing import List, Dict, Any, Tuple
from embedder import Embedder

import math

def compress_score(score: float) -> float:
    return 1 / (1 + math.exp(-5 * (score - 0.7)))

class PyramidRetriever:
    def __init__(self, embedder: Embedder, pyramid_nodes: List[Dict[str, Any]]):
        self.embedder = embedder
        self.pyramid_nodes = pyramid_nodes
        # Stack embeddings into a single matrix for fast batch cosine similarity
        self.node_embeddings = np.vstack([node["embedding"] for node in pyramid_nodes])

    def detect_query_intent(self, query: str) -> str:
        """
        Simple intent classifier to determine which layer might be best suited.
        In a production system, this could be an LLM call.
        """
        query = query.lower()
        if "risk" in query or "compliance" in query or "category" in query or "type of" in query:
            return "Layer 3: Category"
        elif any(word in query for word in ["how", "why", "strategy", "position", "analyze"]):
            return "Layer 2: Summary"
        elif any(word in query for word in ["what was", "how much", "number", "revenue", "who"]):
            return "Layer 4: Distilled Keywords"
        return None # No specific boost

    def search(self, query: str, top_k: int = 3, fuse_scores: bool = True) -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed([query])
        
        # Compute baseline cosine similarities across EVERYTHING
        scores = self.embedder.cosine_similarity(query_emb, self.node_embeddings)
        
        # Apply intent boosting
        preferred_layer = self.detect_query_intent(query)
        
        scored_nodes = []
        for i, node in enumerate(self.pyramid_nodes):
            score = float(scores[i])
            
            # Boost score heavily if it matches the preferred layer for this query
            if preferred_layer and node["layer"] == preferred_layer:
                score *= 2.0  # 200% boost to force intent routing to win
                
            scored_nodes.append({
                "score": score,
                "node": node
            })
            
        if fuse_scores:
            # Reciprocal Rank Fusion / Score Fusion logic
            # If the same chunk_id scores high across multiple layers, combine their evidence
            chunk_scores = collections.defaultdict(list)
            for sn in scored_nodes:
                chunk_scores[sn["node"]["chunk_id"]].append((sn["score"], sn["node"]))
                
            fused_results = []
            for chunk_id, nodes_list in chunk_scores.items():
                # Sort nodes in this chunk by score
                nodes_list.sort(key=lambda x: x[0], reverse=True)
                top_score, top_node = nodes_list[0]
                
                # If there is supporting evidence from other layers above a threshold, boost the fused score
                supporting_evidence = [s for s, n in nodes_list[1:] if s > 0.4] 
                if supporting_evidence:
                    fusion_boost = len(supporting_evidence) * 0.05
                    top_score += fusion_boost
                    
                fused_results.append({
                    "score": round(compress_score(top_score), 2),
                    "node": top_node,
                    "fusion_applied": len(supporting_evidence) > 0
                })
                
            scored_nodes = fused_results

        # Sort final results
        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        return scored_nodes[:top_k]
