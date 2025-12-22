from fastapi import Request, HTTPException

def get_redis_client(request: Request):
    if not hasattr(request.app.state, "redis_client"):
        raise HTTPException(status_code=500, detail="Redis client not initialized.")
    return request.app.state.redis_client