from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional
import replicate
import os
import io
import base64
import tempfile
import aiofiles
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="AI Image Enhancer API",
    description="Enhance images using Real-ESRGAN via Replicate API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Replicate API token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN environment variable is required")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Replicate model
MODEL_NAME = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"

# Request models
class ImageEnhanceRequest(BaseModel):
    image_url: HttpUrl
    scale: Optional[int] = 2
    
class ImageEnhanceResponse(BaseModel):
    enhanced_image_url: str
    original_image_url: str
    scale: int
    status: str

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "AI Image Enhancer API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Image Enhancer"}

# Enhance image from URL
@app.post("/enhance-from-url", response_model=ImageEnhanceResponse)
async def enhance_image_from_url(request: ImageEnhanceRequest):
    """
    Enhance an image from a URL using Real-ESRGAN
    """
    try:
        # Validate scale parameter
        if request.scale not in [2, 4, 8]:
            raise HTTPException(status_code=400, detail="Scale must be 2, 4, or 8")
        
        # Prepare input for Replicate
        input_data = {
            "image": str(request.image_url),
            "scale": request.scale
        }
        
        # Run the model
        output = replicate.run(MODEL_NAME, input=input_data)
        
        # Return response
        return ImageEnhanceResponse(
            enhanced_image_url=str(output),
            original_image_url=str(request.image_url),
            scale=request.scale,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

# Enhance image from file upload
@app.post("/enhance-from-file")
async def enhance_image_from_file(
    file: UploadFile = File(...),
    scale: int = Form(default=2)
):
    """
    Enhance an uploaded image file using Real-ESRGAN
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate scale parameter
        if scale not in [2, 4, 8]:
            raise HTTPException(status_code=400, detail="Scale must be 2, 4, or 8")
        
        # Read file content
        file_content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Prepare input for Replicate (using file path)
            with open(temp_file_path, "rb") as f:
                input_data = {
                    "image": f,
                    "scale": scale
                }
                
                # Run the model
                output = replicate.run(MODEL_NAME, input=input_data)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Return response
            return {
                "enhanced_image_url": str(output),
                "original_filename": file.filename,
                "scale": scale,
                "status": "success"
            }
            
        except Exception as e:
            # Clean up temp file in case of error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

# Download enhanced image
@app.get("/download/{image_id}")
async def download_image(image_id: str):
    """
    Download an enhanced image (this would typically fetch from your storage)
    For now, this is a placeholder that would integrate with your file storage solution
    """
    # This is a placeholder - implement based on how you store enhanced images
    raise HTTPException(status_code=501, detail="Download functionality not implemented")

# Get enhancement status (for async operations)
@app.get("/status/{job_id}")
async def get_enhancement_status(job_id: str):
    """
    Check the status of an enhancement job
    This would be useful if you implement async processing
    """
    # This is a placeholder for async job status checking
    raise HTTPException(status_code=501, detail="Status checking not implemented")

# Batch enhance multiple images
@app.post("/enhance-batch")
async def enhance_batch(
    image_urls: list[HttpUrl],
    scale: int = 2
):
    """
    Enhance multiple images in batch
    """
    try:
        if len(image_urls) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size limited to 10 images")
        
        if scale not in [2, 4, 8]:
            raise HTTPException(status_code=400, detail="Scale must be 2, 4, or 8")
        
        results = []
        errors = []
        
        for i, image_url in enumerate(image_urls):
            try:
                input_data = {
                    "image": str(image_url),
                    "scale": scale
                }
                
                output = replicate.run(MODEL_NAME, input=input_data)
                
                results.append({
                    "index": i,
                    "original_url": str(image_url),
                    "enhanced_url": str(output),
                    "status": "success"
                })
                
            except Exception as e:
                errors.append({
                    "index": i,
                    "original_url": str(image_url),
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "results": results,
            "errors": errors,
            "total_processed": len(results),
            "total_failed": len(errors),
            "scale": scale
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch enhancement failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)