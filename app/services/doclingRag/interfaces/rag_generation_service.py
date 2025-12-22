from typing import Dict, Any

class IRagGenerationService:
    async def generate_answer(
        self,
        question: str,
        context: str,
    ) -> Dict[str, Any]:
        """Generate structured RAG answer"""
        pass
