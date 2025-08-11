#!/usr/bin/env python3
"""
Configuration management for Vector DB23ai MCP RAG Agent
Supports environment variables and secure credential management
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class MCPConfig:
    """MCP Server Configuration"""
    sqlcl_path: str = field(default_factory=lambda: os.getenv('MCP_SQLCL_PATH', '/home/ubuntu/adk-rag-mcp/sqlcl/bin/sql'))
    tns_admin: str = field(default_factory=lambda: os.getenv('TNS_ADMIN', '/home/ubuntu/wallet-db'))
    connection_name: str = field(default_factory=lambda: os.getenv('MCP_CONNECTION_NAME', 'mcp_saved'))
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv('MCP_TIMEOUT', '30')))

@dataclass
class VectorConfig:
    """Vector Search Configuration"""
    embedding_model: str = field(default_factory=lambda: os.getenv('VECTOR_EMBEDDING_MODEL', 'cohere.embed-english-v3.0'))
    vector_dimensions: int = field(default_factory=lambda: int(os.getenv('VECTOR_DIMENSIONS', '1024')))
    distance_metric: str = field(default_factory=lambda: os.getenv('VECTOR_DISTANCE_METRIC', 'COSINE'))
    default_top_k: int = field(default_factory=lambda: int(os.getenv('VECTOR_DEFAULT_TOP_K', '3')))
    default_similarity_threshold: float = field(default_factory=lambda: float(os.getenv('VECTOR_DEFAULT_THRESHOLD', '0.6')))
    vector_function_name: str = field(default_factory=lambda: os.getenv('VECTOR_FUNCTION_NAME', 'oci_cohere_rag_search_fixed'))

@dataclass
class OCIConfig:
    """OCI GenAI Agent Configuration"""
    agent_endpoint_id: Optional[str] = field(default_factory=lambda: os.getenv('OCI_AGENT_ENDPOINT_ID'))
    compartment_id: Optional[str] = field(default_factory=lambda: os.getenv('OCI_COMPARTMENT_ID'))
    region: str = field(default_factory=lambda: os.getenv('OCI_REGION', 'us-ashburn-1'))
    auth_profile: str = field(default_factory=lambda: os.getenv('OCI_AUTH_PROFILE', 'DEFAULT'))
    auth_type: str = field(default_factory=lambda: os.getenv('OCI_AUTH_TYPE', 'api_key'))

@dataclass
class SecurityConfig:
    """Security Configuration"""
    max_query_length: int = field(default_factory=lambda: int(os.getenv('MAX_QUERY_LENGTH', '1000')))
    enable_query_logging: bool = field(default_factory=lambda: os.getenv('ENABLE_QUERY_LOGGING', 'false').lower() == 'true')

@dataclass
class AppConfig:
    """Complete Application Configuration"""
    mcp: MCPConfig = field(default_factory=MCPConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    oci: OCIConfig = field(default_factory=OCIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    environment: str = field(default_factory=lambda: os.getenv('APP_ENVIRONMENT', 'development'))
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')

def load_config() -> AppConfig:
    """Load configuration from environment variables"""
    return AppConfig()

def get_sql_injection_safe_query(query: str, max_length: int = 1000) -> str:
    """Sanitize query for SQL injection protection"""
    # Truncate if too long
    if len(query) > max_length:
        query = query[:max_length]
    
    # Escape single quotes for SQL safety
    safe_query = query.replace("'", "''")
    
    return safe_query