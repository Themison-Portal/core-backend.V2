import uuid
from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

# IMPORTANT: Import the Vector type from pgvector
from pgvector.sqlalchemy import Vector 

# Assuming Base is imported from your database setup file
from database import Base 

# Define the dimensions for your embedding model (e.g., 1536 for OpenAI)
# EMBEDDING_DIMENSION = 1536
EMBEDDING_DIMENSION = 768  # For BioBERT

class ProtocolChunk(Base):
    __tablename__ = "protocol_chunks"

    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4, 
        index=True
    )
    
    # 1. The Vector Column: Stores the numerical representation of the text
    # The number 1536 is the dimension/size of the vector array.
    embedding = Column(Vector(EMBEDDING_DIMENSION))
    
    # 2. The Text Content: Stores the original chunk of text
    content = Column(Text, nullable=False)
    
    # 3. Metadata: Links back to the original document/protocol
    protocol_id = Column(UUID(as_uuid=True), ForeignKey("protocols.id"), nullable=False)

    # 4. Page Number
    page_number = Column(Integer, nullable=True)

    # 5. Paragraph Number
    paragraph_number = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Optional relationship definition (if you define a Protocol model)
    protocol = relationship("Protocol", back_populates="chunks")

# You might also need a simple model for the Protocols themselves:
class Protocol(Base):
    __tablename__ = "protocols"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    title = Column(String, nullable=False)

    
    # Relationship to chunks (if using relationship)
    chunks = relationship("ProtocolChunk", back_populates="protocol")