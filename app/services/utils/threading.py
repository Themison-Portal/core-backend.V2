import asyncio
from functools import partial
async def run_in_thread(fn, *args, **kwargs):
    """Utility to run blocking synchronous functions in a dedicated thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))