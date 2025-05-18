from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import shutil
from agents.main_agent import MainAgent
from processors.data_processor import DataProcessor
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Initialize FastAPI app
app = FastAPI(title="RAG Application")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directory if it doesn't exist
Path("data").mkdir(exist_ok=True)

# Initialize components
data_processor = DataProcessor(data_dir="data")
main_agent = MainAgent(data_processor=data_processor)

class VisualizationRequest(BaseModel):
    dataset_name: str
    plot_type: str
    x_column: str
    y_column: str
    title: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint to verify server is running."""
    return {"status": "ok", "message": "Server is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/datasets")
async def list_datasets():
    """List all available datasets."""
    return {"datasets": data_processor.list_datasets()}

@app.get("/dataset/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    """Get information about a specific dataset."""
    return data_processor.get_dataset_info(dataset_name)

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file."""
    try:
        logger.info(f"Receiving file upload: {file.filename}")
        file_path = f"data/{file.filename}"
        logger.info(f"Saving file to: {file_path}")
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load the dataset
        if data_processor.load_dataset(file.filename):
            return {"message": f"Successfully uploaded and loaded {file.filename}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to load dataset")
            
    except Exception as e:
        logger.error(f"Error in upload_csv: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def process_query(query: str):
    """Process a query using the main agent."""
    try:
        logger.info(f"Received query: {query}")
        
        # Check if any dataset is loaded
        datasets = data_processor.list_datasets()
        if not datasets:
            logger.error("No dataset loaded")
            raise HTTPException(
                status_code=400,
                detail="Please upload a dataset first"
            )
        
        # Process the query with error catching
        try:
            result = main_agent.process_query(query)
            logger.info(f"Query processed with status: {result.get('status', 'unknown')}")
            
            # Handle different result statuses
            if result.get("status") == "error":
                logger.error(f"Query processing error: {result.get('error')}")
                return {
                    "status": "error",
                    "error": result.get("error", "Unknown error occurred"),
                    "query": query
                }
            elif result.get("status") == "timeout":
                logger.warning("Query processing timed out")
                return {
                    "status": "timeout",
                    "error": "Query processing timed out",
                    "query": query
                }
            
            # Successful response
            return {
                "status": "success",
                "response": result.get("response", ""),
                "raw_llm_response": result.get("raw_llm_response", ""),
                "confidence": result.get("confidence", 0.0),
                "query_type": result.get("query_type", "unknown"),
                "sample_size": result.get("sample_size", 0)
            }
            
        except Exception as query_error:
            logger.exception("Error in query processing")
            return {
                "status": "error",
                "error": str(query_error),
                "query": query
            }
            
    except Exception as e:
        logger.exception("Unexpected error in query endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {str(e)}"
        )

@app.post("/visualize")
async def create_visualization(request: VisualizationRequest):
    """Create a visualization from dataset."""
    try:
        # Get the dataset
        df = data_processor.get_dataset(request.dataset_name)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' not found")

        # Create figure
        plt.figure(figsize=(10, 6))
        
        if request.plot_type == "bar":
            plt.bar(df[request.x_column], df[request.y_column])
        elif request.plot_type == "line":
            plt.plot(df[request.x_column], df[request.y_column])
        elif request.plot_type == "scatter":
            plt.scatter(df[request.x_column], df[request.y_column])
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported plot type: {request.plot_type}")

        # Set labels and title
        plt.xlabel(request.x_column)
        plt.ylabel(request.y_column)
        plt.title(request.title or f"{request.plot_type.capitalize()} Plot")
        plt.xticks(rotation=45)
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode the image to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        
        return {
            "image": img_str,
            "plot_type": request.plot_type,
            "title": request.title
        }
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8080,
            log_level="info",
            reload=False  # Disable reload to avoid duplicate process
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise 