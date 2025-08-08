"""
Supabase client
"""

import os
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


class SupabaseClient:
    """Singleton class for Supabase client"""
    
    _instance: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Optional[Client]:
        """Get the Supabase client singleton"""
        if cls._instance is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_ANON_KEY")
            cls._instance = create_client(url, key)
        return cls._instance


def supabase_client() -> Optional[Client]:
    """Get the Supabase client"""
    return SupabaseClient.get_client()
    