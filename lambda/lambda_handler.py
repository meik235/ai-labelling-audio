"""AWS Lambda handler for Label Studio ML Backend."""

from mangum import Mangum
from backend.app import app

# Create Mangum adapter for FastAPI app
handler = Mangum(app, lifespan="off")

# Lambda entry point
def lambda_handler(event, context):
    """
    AWS Lambda handler entry point.
    
    This wraps the FastAPI app using Mangum, which converts
    Lambda events to ASGI requests and ASGI responses back to Lambda.
    
    Args:
        event: Lambda event (API Gateway or ALB event)
        context: Lambda context object
        
    Returns:
        API Gateway/ALB compatible response
    """
    return handler(event, context)

