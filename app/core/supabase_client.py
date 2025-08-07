import os
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

supabase_instance: Optional[Client] = None

def supabase_client() -> Optional[Client]:
    global supabase_instance
    
    if supabase_instance is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        supabase_instance = create_client(url, key)
    
    return supabase_instance
    