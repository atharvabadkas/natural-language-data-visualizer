from typing import Dict, List, Optional, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
from functools import lru_cache
from processors.data_processor import DataProcessor

class QueryType:
    METADATA = "metadata"
    FACTUAL = "factual"
    AGGREGATION = "aggregation"
    TIME_SERIES = "time_series"
    COMPARISON = "comparison"

class MainAgent:
    def __init__(
        self,
        model_name: str = "llama2:latest",
        temperature: float = 0.1,
        data_processor: Optional[DataProcessor] = None,
        cache_dir: str = ".cache"
    ):
        self._setup_logging()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._initialize_llm(model_name, temperature)
        self._initialize_embeddings()
        self.data_processor = data_processor or DataProcessor(data_dir="data")
        
        # Setup data structures
        self.df = None
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # Reduced for faster processing
            chunk_overlap=50,    # Minimal overlap for timestamp-based data
            separators=["\n", ","],  # CSV-specific separators
            keep_separator=False,
            strip_whitespace=True
        )
        
        # Add timestamp index for faster querying
        self.time_index = None
        
        self._setup_prompts()
        self._setup_cache()
    
    def _setup_cache(self):
        """Setup simple in-memory cache for queries."""
        self.query_cache = {}
        self.cache_dir.mkdir(exist_ok=True)
    
    def _initialize_embeddings(self):
        """Initialize embedding models for semantic search with re-ranking."""
        # Primary embedding model for initial retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Better semantic understanding
            cache_folder=str(self.cache_dir / "embeddings"),
            encode_kwargs={'normalize_embeddings': True}  # Improved vector quality
        )
        
        # Initialize cross-encoder for re-ranking
        from sentence_transformers import CrossEncoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _classify_query(self, query: str) -> str:
        """Classify query type for optimized processing."""
        query_lower = query.lower()
        
        # Simple rule-based classification
        if any(word in query_lower for word in ["rows", "columns", "shape", "structure"]):
            return QueryType.METADATA
        elif any(word in query_lower for word in ["average", "sum", "total", "mean", "count"]):
            return QueryType.AGGREGATION
        elif any(word in query_lower for word in ["trend", "over time", "pattern", "timeline"]):
            return QueryType.TIME_SERIES
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return QueryType.COMPARISON
        return QueryType.FACTUAL

    def _create_vector_store(self):
        """Create vector store from dataset with chunking."""
        if self.df is None:
            return
        
        # Convert DataFrame to documents
        documents = []
        for idx, row in self.df.iterrows():
            doc = " ".join([f"{col}: {row[col]}" for col in self.df.columns])
            documents.append(doc)
        
        # Split into chunks for large datasets
        chunks = self.text_splitter.split_text(" ".join(documents))
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            chunks,
            self.embeddings,
            metadatas=[{"chunk_id": i} for i in range(len(chunks))]
        )

    @lru_cache(maxsize=100)
    def _semantic_search(self, query: str, k: int = 10) -> List[Dict]:
        """Perform semantic search with re-ranking for better accuracy."""
        if self.vector_store is None:
            self._create_vector_store()
        
        # Initial retrieval with FAISS
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Prepare pairs for re-ranking
        pairs = [(query, doc.page_content) for doc, _ in results]
        
        # Re-rank using cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Combine results with new scores and sort
        ranked_results = [
            {
                "content": doc.page_content,
                "score": float(score),  # Convert to float for JSON serialization
                "metadata": doc.metadata
            }
            for (doc, _), score in zip(results, scores)
        ]
        
        # Sort by cross-encoder score and take top 5
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        return ranked_results[:5]

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for faster querying with optimized indexing."""
        # Convert timestamp to datetime and create time-based indices
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Calculate net weights and round to 2 decimals
        df['net_weight'] = (df['total weight'] - df['vessel weight']).round(2)
        
        # Create categorical indices for faster grouping
        for col in ['vessel name', 'ingredient sku']:
            df[f'{col}_idx'] = pd.Categorical(df[col]).codes
        
        # Create summary text for better semantic search
        df['search_text'] = df.apply(
            lambda row: f"At {row['timestamp']} used {row['ingredient sku']} "
                      f"in {row['vessel name']} with net weight {row['net_weight']}g",
            axis=1
        )
        
        # Sort by timestamp for time-series operations
        df.sort_values('timestamp', inplace=True)
        
        # Create indices for faster lookups
        df.set_index(['timestamp', 'vessel name', 'ingredient sku'], inplace=True)
        df.sort_index(inplace=True)
        
        return df

    def _get_relevant_data(self, query: str, query_type: str) -> pd.DataFrame:
        """Get relevant data with simple caching."""
        cache_key = f"{query_type}_{hash(query)}"
        
        # Check cache
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        try:
            # Get base data
            if query_type == QueryType.TIME_SERIES:
                relevant_data = self.df.sort_values('timestamp')
            elif query_type == QueryType.AGGREGATION:
                # Handle aggregation based on available columns
                group_cols = ['vessel name', 'ingredient sku'] if 'ingredient sku' in self.df.columns else ['vessel name']
                agg_cols = ['net_weight'] if 'net_weight' in self.df.columns else ['total weight']
                relevant_data = self.df.groupby(group_cols)[agg_cols].agg(['sum', 'mean', 'count']).reset_index()
            elif query_type == QueryType.METADATA:
                if "vessel" in query.lower():
                    # Get vessel information
                    vessel_data = self.df.groupby('vessel name').agg({
                        'vessel weight': 'first',
                        'total weight': ['count', 'sum', 'mean']
                    }).reset_index()
                    vessel_data.columns = ['vessel_name', 'vessel_weight', 'usage_count', 'total_weight_sum', 'total_weight_mean']
                    relevant_data = vessel_data
                elif "ingredient" in query.lower():
                    # Get ingredient information
                    if 'ingredient sku' in self.df.columns:
                        ingredient_data = self.df.groupby('ingredient sku').agg({
                            'net_weight' if 'net_weight' in self.df.columns else 'total weight': ['count', 'sum', 'mean']
                        }).reset_index()
                        relevant_data = ingredient_data
                    else:
                        relevant_data = self.df.head()
                else:
                    relevant_data = self.df.head()
            else:
                # Use semantic search for other query types
                results = self._semantic_search(query)
                relevant_indices = [int(res['metadata']['chunk_id']) for res in results]
                relevant_data = self.df.iloc[relevant_indices]
            
            # Cache the results
            self.query_cache[cache_key] = relevant_data
            return relevant_data
            
        except Exception as e:
            self.logger.error(f"Error getting relevant data: {str(e)}")
            # Return a small sample of the data if there's an error
            return self.df.head()

    def _verify_answer(self, response: str, data_sample: pd.DataFrame) -> Dict[str, Any]:
        """Verify answer against data and add confidence score."""
        # Extract numbers and key phrases from response
        numbers_in_response = [float(s) for s in response.split() if s.replace('.','',1).isdigit()]
        
        # Check if numbers appear in data
        confidence = 0.0
        for num in numbers_in_response:
            if any(num in row.values for _, row in data_sample.iterrows()):
                confidence += 0.5
        
        # Add base confidence
        confidence = max(0.1, min(confidence + 0.3, 1.0))
        
        return {
            "response": response,
            "confidence": confidence,
            "verified": confidence > 0.5
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query processing with timeout handling."""
        try:
            # Load dataset if needed
            if self.df is None:
                datasets = self.data_processor.list_datasets()
                if not datasets:
                    return {"error": "No dataset loaded", "status": "error"}
                self.df = self.data_processor.get_dataset(datasets[0])
                self._create_vector_store()
            
            # Classify query
            query_type = self._classify_query(query)
            
            # Get relevant data
            relevant_data = self._get_relevant_data(query, query_type)
            
            # Prepare context
            data_context = {
                "columns": list(self.df.columns),
                "total_rows": len(self.df),
                "query": query,
                "data_sample": relevant_data.to_dict('records')[:5],  # Limit sample size
                "query_type": query_type
            }
            
            try:
                # Process query with timeout handling
                chain = self.query_prompt | self.llm
                response = chain.invoke(data_context)
                
                # Enhanced logging
                self.logger.info("=" * 50)
                self.logger.info("Query: %s", query)
                self.logger.info("Query Type: %s", query_type)
                self.logger.info("Raw LLM Response:")
                self.logger.info(response.content)
                self.logger.info("=" * 50)
                
                # Verify and add confidence
                verified_response = self._verify_answer(response.content.strip(), relevant_data)
                
                return {
                    "raw_llm_response": response.content,
                    "response": verified_response["response"],
                    "confidence": verified_response["confidence"],
                    "verified": verified_response["verified"],
                    "query_type": query_type,
                    "sample_size": len(relevant_data),
                    "status": "success"
                }
            
            except TimeoutError:
                self.logger.error("LLM response timed out")
                return {
                    "error": "Query processing timed out. Please try again with a simpler query.",
                    "status": "timeout",
                    "query": query
                }

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "error": str(e),
                "status": "error",
                "query": query
            }

    def _setup_prompts(self):
        """Setup optimized prompts for restaurant data processing."""
        self.query_prompt = ChatPromptTemplate.from_template(
            """You are a data analyst for a restaurant kitchen. Answer questions precisely using the provided data.

            Current Question: {query}
            Query Type: {query_type}
            
            Data Context:
            - Total Rows: {total_rows}
            - Columns: {columns}
            
            Available Data:
            {data_sample}
            
            Instructions for {query_type} query:
            
            For vessel-related queries:
            - List each vessel name
            - Show usage count for each vessel
            - Include vessel weight
            - Format: "1. [Vessel Name] (used X times, weight: Y g)"
            
            For ingredient-related queries:
            - List each ingredient
            - Show usage count
            - Include total weight used
            - Format: "1. [Ingredient] (used X times, total weight: Y g)"
            
            For other queries:
            - Provide exact numbers from data
            - Include all relevant measurements
            - Show calculations if needed
            
            Response must be complete and accurate. Use exact values from the data."""
        )

    def _setup_logging(self):
        """Setup basic logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _initialize_llm(self, model_name: str, temperature: float):
        """Initialize LLM with optimized settings to prevent timeouts."""
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=2048,         # Reduced context window
            num_thread=4,         # Balanced thread usage
            num_gpu=1,            # GPU acceleration
            top_k=10,             # Reduced for faster responses
            top_p=0.7,            # More focused sampling
            repeat_penalty=1.1,    # Standard penalty
            num_predict=512,       # Balanced response length
            stop=["\n\nHuman:", "\n\nAssistant:"],
            timeout=120,          # 2-minute timeout
            verbose=True          # Enable verbose mode for debugging
        )
        