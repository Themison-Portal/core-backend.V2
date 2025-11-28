"""
Supabase client
"""

import os
import sys
import logging
import threading
from typing import Optional
from app.config import get_settings
# from dotenv import load_dotenv
from supabase import Client, create_client

# load_dotenv()


class SupabaseClient:
    """Singleton class for Supabase client"""
    
    _instance: Optional[Client] = None
    _lock = threading.Lock()

    @classmethod
    def get_client(cls) -> Optional[Client]:
        """Get the Supabase client singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        # url = os.getenv("SUPABASE_URL")
                        # key = os.getenv("SUPABASE_ANON_KEY")
                        settings = get_settings()    
                        if not settings:
                            logging.error("Settings could not be loaded.")
                            raise RuntimeError("Settings could not be loaded.")                    
                        url = settings.supabase_url
                        key = settings.supabase_service_key

                        if not url or not key:
                            logging.error("Supabase URL or key not set in environment variables.")
                            raise ValueError("Supabase URL or key not set in environment variables.")

                        cls._instance = create_client(url, key)
                    except Exception as e:
                        logging.error(f"Failed to create Supabase client: {e}")
                        return None
                        
        return cls._instance


def supabase_client() -> Optional[Client]:
    """Get the Supabase client"""
    return SupabaseClient.get_client()

