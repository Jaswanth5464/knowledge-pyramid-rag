"""
The core of the Knowledge Pyramid.
Instead of dealing with raw document chunks at query time, this module performs
"agentic knowledge distillation" during ingestion. It extracts high-value signal
into different abstraction layers: Summaries (context), Categories (topics), 
and Keywords (facts).
"""

import collections
import re
from typing import List, Dict, Any, Tuple

# Simple stopwords for basic keyword extraction (Layer 4)
STOP_WORDS = set([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
    "about", "against", "between", "into", "through", "during", "before", "after", 
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
    "under", "again", "further", "then", "once", "here", "there", "when", "where", 
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", 
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "is", "of",
    "by", "as", "it", "that", "was", "were", "be", "this", "are", "have", "has", "had",
    "our", "we", "they", "their", "its", "your", "my", "nearly", "significantly", 
    "exclusively", "approximately", "per", "during", "quarter", "company", "year"
])

# Rule-based categories (Layer 3)
CATEGORIES = {
    "Finance & Revenue": ["revenue", "profit", "earnings", "billion", "growth", "margin", "quarterly", "financial", "shareholder", "sales"],
    "Technology & AI": ["ai", "model", "cloud", "neural", "software", "infrastructure", "compute", "innovation", "technology", "artificial", "intelligence", "generation"],
    "Risk & Compliance": ["risk", "compliance", "audit", "regulation", "law", "lawsuit", "threat", "security", "breach", "legal"]
}


class DocumentChunker:
    @staticmethod
    def sliding_window(pages: List[str], window_size: int = 2) -> List[Tuple[int, int, str]]:
        """
        Creates sliding windows to prevent context loss across page boundaries.
        Returns: [(start_page_idx, end_page_idx, text_chunk)]
        """
        if not pages:
            return []
        
        if len(pages) < window_size:
            return [(0, len(pages)-1, "\n".join(pages))]
            
        chunks = []
        for i in range(len(pages) - window_size + 1):
            chunk_text = "\n".join(pages[i:i+window_size])
            chunks.append((i, i+window_size-1, chunk_text))
            
        return chunks


class KnowledgePyramidBuilder:
    def __init__(self, embedder):
        self.embedder = embedder

    def distill_keywords(self, text: str, top_k: int = 12) -> str:
        """Layer 4: Maximum compression fact layer."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        meaningful_words = [w for w in words if w not in STOP_WORDS]
        
        # Simple TF calculation (ignoring IDF for this assignment context)
        # We also boost slightly longer words as they tend to be more specific entities
        word_scores = collections.defaultdict(float)
        for w in meaningful_words:
            word_scores[w] += 1.0 + (len(w) * 0.01)
            
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ", ".join([w[0] for w in top_words])

    def categorize_chunk(self, text: str) -> str:
        """Layer 3: Thematic signal layer (Rule-based)."""
        text_lower = text.lower()
        best_category = "General Operation"
        best_score = 0
        
        for category, keywords in CATEGORIES.items():
            score = sum(text_lower.count(k) for k in keywords)
            if score > best_score:
                best_score = score
                best_category = category
                
        return best_category

    def summarize_chunk(self, text: str) -> str:
        """Layer 2: Context-aware compression layer."""
        sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
        if not sentences:
            return ""
        summary = sentences[0]
        for s in sentences[1:]:
            if len(summary) + len(s) < 250:
                summary += " " + s
            else:
                break
        return summary

    def build_pyramid(self, document_pages: List[str]) -> List[Dict[str, Any]]:
        """
        Ingests document pages, chunks them, and builds a multi-layer semantic pyramid.
        Returns a flat list of nodes (to be easily searchable), each bearing layer metadata.
        """
        print("[Pyramid] Building 4-layer Knowledge Pyramid from document pages...")
        chunks = DocumentChunker.sliding_window(document_pages, window_size=2)
        
        pyramid_nodes = []
        texts_to_embed = []
        
        for start_idx, end_idx, chunk_text in chunks:
            # Build layers
            raw_text = chunk_text
            summary = self.summarize_chunk(chunk_text)
            category_name = self.categorize_chunk(chunk_text)
            keywords = self.distill_keywords(chunk_text)
            category_text = f"[{category_name}] {keywords}" # use keywords for more distinct content
            
            # Prepare nodes
            base_meta = {
                "pages": f"{start_idx+1}-{end_idx+1}",
                "chunk_id": start_idx,  # unique ID for the chunk this came from
                "raw_text": raw_text    # Store original text for downstream LLM generation
            }
            
            # Node: Raw (Layer 1)
            pyramid_nodes.append({**base_meta, "layer": "Layer 1: Raw Text", "text": raw_text})
            texts_to_embed.append(raw_text)
            
            # Node: Summary (Layer 2)
            pyramid_nodes.append({**base_meta, "layer": "Layer 2: Summary", "text": summary})
            texts_to_embed.append(summary)
            
            # Node: Category (Layer 3)
            pyramid_nodes.append({**base_meta, "layer": "Layer 3: Category", "text": category_text, "category_name": category_name})
            texts_to_embed.append(category_text)
            
            # Node: Keywords (Layer 4)
            pyramid_nodes.append({**base_meta, "layer": "Layer 4: Distilled Keywords", "text": keywords})
            texts_to_embed.append(keywords)

        # Batch embed all nodes for efficiency
        print(f"[Pyramid] Embedding {len(texts_to_embed)} nodes across {len(chunks)} chunks...")
        embeddings = self.embedder.embed(texts_to_embed)
        
        for i, node in enumerate(pyramid_nodes):
            node["embedding"] = embeddings[i]
            
        print("[Pyramid] Pyramid complete.")
        return pyramid_nodes
