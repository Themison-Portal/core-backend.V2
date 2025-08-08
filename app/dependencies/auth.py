"""
This module contains the authentication dependencies.
"""
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.supabase_client import supabase_client

"""
Create a Supabase client.
"""
supabase = supabase_client()
security = HTTPBearer()

"""
Dependency to get current authenticated user information.
"""
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user information.
    Use this when you need access to user data in your route.
    """
    token = credentials.credentials
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Return user data in a consistent format
        user = user_response.user
        return {
            "id": user.id,
            "email": user.email,
            "user_metadata": user.user_metadata or {},
            "app_metadata": user.app_metadata or {},
            "aud": user.aud,
            "created_at": user.created_at,
            "updated_at": user.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Failed to get user information: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

"""
Dependency to get just the current user ID.
"""
async def get_current_user_id(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> str:
    """
    Dependency to get just the current user ID.
    Use this when you only need the user ID for operations.
    """
    return current_user["id"]

# Legacy support - remove once migration is complete
"""
Legacy support - remove once migration is complete.
"""
class AuthDependency:
    @staticmethod
    async def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Legacy method - use verify_bearer_token or get_current_user instead"""
        # This maintains backward compatibility
        return await get_current_user(credentials)

auth = AuthDependency()