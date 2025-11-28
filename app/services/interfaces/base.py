"""
Base interface for all services
"""

from typing import Generic, List, TypeVar
from uuid import UUID

from app.contracts.base import BaseContract

CreateContractType = TypeVar("create_contract_type", bound=BaseContract)
UpdateContractType = TypeVar("update_contract_type", bound=BaseContract)
ResponseContractType = TypeVar("response_contract_type", bound=BaseContract)

class IBaseService(Generic[CreateContractType, UpdateContractType, ResponseContractType]):
    """
    Base interface for all services
    """

    async def create(self, contract: CreateContractType) -> ResponseContractType:
        """
        Create a new record
        """
        pass
    
    async def get(self, id: UUID) -> ResponseContractType:
        """
        Get a record by ID
        """
        pass
    
    async def update(self, id: UUID, contract: UpdateContractType) -> ResponseContractType:
        """
        Update a record by ID
        """
        pass
    
    async def delete(self, id: UUID) -> None:
        """
        Delete a record by ID
        """
        pass
    
    async def list(self) -> List[ResponseContractType]:
        """
        List all records
        """
        pass 