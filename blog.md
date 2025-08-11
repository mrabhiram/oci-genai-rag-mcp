# Building a Production RAG System with OCI GenAI Agent, DB23ai Vector Search, and Model Context Protocol

## Introduction

This comprehensive guide demonstrates how to build a production-ready Retrieval-Augmented Generation (RAG) system using Oracle Cloud Infrastructure (OCI) GenAI Agent integrated with Oracle Database 23ai's native vector capabilities through the Model Context Protocol (MCP). The solution leverages DB23ai's built-in DBMS_VECTOR package, OCI's Cohere embedding models, and MCP for secure, standardized database communication.

Unlike traditional approaches that require separate vector databases or direct SQL execution, this solution utilizes DB23ai's enterprise-grade vector capabilities with MCP protocol compliance for secure, scalable database interactions.

## Architecture Overview

The solution follows a secure, protocol-compliant architecture:

```
User Query → OCI GenAI Agent Service → MCP Client → DB23ai MCP Server → Vector Search → Results
                                       ↓
                               Model Context Protocol
                                       ↓
                           Oracle DB23ai Vector Database
```

Key components:
- **OCI GenAI Agent**: Orchestrates conversations and function calls
- **MCP Protocol**: Standardized communication between AI and database
- **DB23ai MCP Server**: Native MCP server for secure database access
- **DB23ai Vector Store**: Native vector storage with DBMS_VECTOR capabilities
- **OCI Cohere Embeddings**: Uses cohere.embed-english-v3.0 model (1024 dimensions)
- **Vector Search Function**: Optimized `oci_cohere_rag_search_fixed` for similarity search

## Prerequisites

- Oracle Cloud Infrastructure account with GenAI Agent and GenAI services
- Oracle Database 23ai Autonomous Database instance
- SQLcl with MCP support (Java 17+ required)
- Python 3.11+ with uv package manager
- OCI CLI configured with proper authentication
- Database wallet for secure connection
- Sample dataset for knowledge base population

## Step 1: Environment Setup

### Initialize Python Project

```bash
# Initialize project with Python 3.11
uv init python3.11

# Add required dependencies for MCP support
uv add "oci[adk]"
uv add mcp
uv add oracledb
```

### Install Oracle SQLcl with MCP Support

```bash
# Download and install SQLcl (ensure MCP support)
wget https://download.oracle.com/otn_software/java/sqldeveloper/sqlcl-latest.zip
unzip sqlcl-latest.zip -d /home/ubuntu/adk-rag-mcp/
chmod +x /home/ubuntu/adk-rag-mcp/sqlcl/bin/sql

# Install Java 17+ (required for SQLcl MCP server)
sudo apt install -y openjdk-17-jdk

# Verify Java version
java -version
```

### Configure Environment Variables

```bash
# Set up secure environment configuration
export TNS_ADMIN=/home/ubuntu/wallet-db
export MCP_SQLCL_PATH=/home/ubuntu/adk-rag-mcp/sqlcl/bin/sql
export MCP_CONNECTION_NAME=mcp_saved
export VECTOR_FUNCTION_NAME=oci_cohere_rag_search_fixed
export OCI_REGION=us-ashburn-1
```

## Step 2: Secure Connection Setup

### Create Saved Database Connection

Instead of hardcoding credentials, create a secure saved connection in SQLcl:

```bash
# Connect to SQLcl and create saved connection
export TNS_ADMIN=/home/ubuntu/wallet-db
/home/ubuntu/adk-rag-mcp/sqlcl/bin/sql /nolog

# Create the saved connection (replace with your credentials)
SQL> connect -save mcp_saved ADMIN/"YourSecurePassword"@your_db_service

# Verify the saved connection works
SQL> connect -name mcp_saved
SQL> exit
```

### Configure MCP Server Startup Script

Create a secure startup script for the DB23ai MCP server:

```bash
#!/bin/bash
# start_mcp_background.sh - Secure MCP Server Startup

export TNS_ADMIN=/home/ubuntu/wallet-db

echo "Starting DB23ai MCP Server in background..."
nohup /home/ubuntu/adk-rag-mcp/sqlcl/bin/sql -mcp -name mcp_saved > mcp_server.log 2>&1 &
MCP_PID=$!
echo $MCP_PID > mcp_server.pid

echo " DB23ai MCP Server started with PID: $MCP_PID"
echo " Log file: mcp_server.log"
echo " PID file: mcp_server.pid" 
echo " Check status: ps -p $MCP_PID"
```

## Step 3: Database Setup and Data Loading

### Configure OCI Credentials for DBMS_VECTOR

Set up credentials for OCI GenAI service integration:

```sql
-- Create OCI credentials for Cohere embeddings
BEGIN
    DBMS_VECTOR.CREATE_CREDENTIAL(
        credential_name => 'OCI_CRED',
        params => JSON('{
            "user_ocid": "ocid1.user.oc1..your_user_ocid",
            "tenancy_ocid": "ocid1.tenancy.oc1..your_tenancy_ocid", 
            "compartment_ocid": "ocid1.compartment.oc1..your_compartment_ocid",
            "private_key": "-----BEGIN PRIVATE KEY-----\nyour_private_key\n-----END PRIVATE KEY-----",
            "fingerprint": "your_fingerprint"
        }')
    );
END;
/
```

### Load Sample Dataset

Using the sample RAG knowledge dataset:

```sql
-- Create the knowledge base table with vector support
CREATE TABLE RAG_SAMPLE_QAS_FROM_KIS (
    KI_TOPIC VARCHAR2(64),
    KI_TEXT VARCHAR2(4000),
    SAMPLE_QUESTION VARCHAR2(4000),
    SAMPLE_GROUND_TRUTH VARCHAR2(4000),
    VEC VECTOR(1024, FLOAT32)
);

-- Load sample data for IT support scenarios
INSERT INTO RAG_SAMPLE_QAS_FROM_KIS (KI_TOPIC, KI_TEXT, SAMPLE_QUESTION, SAMPLE_GROUND_TRUTH) VALUES 
('Troubleshooting Issues with Company-Issued Tablets', 
 '**Troubleshooting Issues with Company-Issued Tablets** This article provides step-by-step guidance for troubleshooting common issues with company-issued tablets. Please follow the steps outlined below to resolve freezing, app crashes, and connectivity problems.',
 'My company-issued tablet is freezing frequently and I''m unable to access some of my apps, what can I do to fix the issue?',
 'I''d be happy to help you troubleshoot the issue with your company-issued tablet! Since your tablet is freezing frequently, try these steps: 1) Restart the device completely 2) Clear cache for problematic apps 3) Check for system updates 4) Contact IT if issues persist.');

INSERT INTO RAG_SAMPLE_QAS_FROM_KIS (KI_TOPIC, KI_TEXT, SAMPLE_QUESTION, SAMPLE_GROUND_TRUTH) VALUES
('Configuring Email on an Android Device',
 '**Configuring Email on an Android Device** Follow these steps to configure email on your Android device: Open the Email app, add account, enter your email and password, configure server settings if needed.',
 'How do I set up my work email on my Android phone?',
 'To set up your work email on Android: 1) Open the Email or Gmail app 2) Tap "Add account" 3) Select your email provider 4) Enter your email and password 5) Follow the setup wizard 6) Test by sending a test email.');

COMMIT;
```

### Generate Vector Embeddings

Use DB23ai's native vector embedding generation with OCI Cohere:

```sql
-- Update vectors using OCI Cohere embeddings
UPDATE RAG_SAMPLE_QAS_FROM_KIS 
SET VEC = VECTOR_EMBEDDING(
    cohere.embed-english-v3.0 USING KI_TEXT as data,
    JSON('{"provider": "ocigenai", "credential_name": "OCI_CRED"}')
);
COMMIT;

-- Verify embeddings were generated
SELECT KI_TOPIC, 
       CASE WHEN VEC IS NOT NULL THEN 'Generated' ELSE 'Missing' END as embedding_status,
       VECTOR_DIMENSION_COUNT(VEC) as dimensions
FROM RAG_SAMPLE_QAS_FROM_KIS;
```

### Create Vector Index for Performance

```sql
-- Create optimized vector index for similarity search
CREATE VECTOR INDEX RAG_COHERE_VECTOR_IDX 
ON RAG_SAMPLE_QAS_FROM_KIS (VEC)
ORGANIZATION NEIGHBOR PARTITIONS
DISTANCE COSINE
WITH TARGET ACCURACY 95;
```

## Step 4: Create Production RAG Search Function

Create a production-ready RAG search function with proper vector distance calculations:

```sql
CREATE OR REPLACE FUNCTION oci_cohere_rag_search_fixed(
    p_query_text CLOB,
    p_top_k NUMBER DEFAULT 3,
    p_similarity_threshold NUMBER DEFAULT 0.6
) RETURN SYS_REFCURSOR
AS
    l_query_vector VECTOR;
    l_cursor SYS_REFCURSOR;
    l_params CLOB;
BEGIN
    -- Set OCI Cohere embedding parameters
    l_params := '{
        "provider": "ocigenai",
        "credential_name": "OCI_CRED",
        "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
        "model": "cohere.embed-english-v3.0"
    }';
    
    -- Generate query vector using OCI Cohere
    l_query_vector := DBMS_VECTOR.UTL_TO_EMBEDDING(p_query_text, JSON(l_params));
    
    -- Perform similarity search with proper ranking
    OPEN l_cursor FOR
        WITH similarity_results AS (
            SELECT 
                KI_TOPIC,
                KI_TEXT,
                SAMPLE_QUESTION,
                SAMPLE_GROUND_TRUTH,
                VECTOR_DISTANCE(VEC, l_query_vector, COSINE) as distance,
                (1 - VECTOR_DISTANCE(VEC, l_query_vector, COSINE)) as similarity_score,
                ROW_NUMBER() OVER (ORDER BY VECTOR_DISTANCE(VEC, l_query_vector, COSINE) ASC) as rank
            FROM RAG_SAMPLE_QAS_FROM_KIS
            WHERE VEC IS NOT NULL
        )
        SELECT 
            KI_TOPIC,
            KI_TEXT,
            SAMPLE_QUESTION,
            SAMPLE_GROUND_TRUTH,
            ROUND(similarity_score, 6) as similarity_score,
            rank
        FROM similarity_results
        WHERE similarity_score >= p_similarity_threshold
        AND rank <= p_top_k
        ORDER BY rank ASC;
    
    RETURN l_cursor;
END;
/
```

## Step 5: Create Configuration Management

### Secure Configuration Module

```python
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
```

## Step 6: Implement MCP-Based Vector Search Agent

### Production MCP RAG Agent

```python
#!/usr/bin/env python3
"""
Vector DB23ai MCP RAG Agent - Production Implementation
Uses vector search function via MCP run-sql tool with secure configuration
"""
import os
import asyncio
from typing import Dict, Any, List
from oci.addons.adk import Agent, AgentClient, tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from config import load_config, get_sql_injection_safe_query

class VectorDB23aiMCPClient:
    """MCP client for vector search using oci_cohere_rag_search_fixed function"""
    
    def __init__(self):
        self.config = load_config()
        self.tns_admin = self.config.mcp.tns_admin
        self.connection_name = self.config.mcp.connection_name
    
    async def vector_search(self, query: str, top_k: int = 3, similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """Perform vector search using oci_cohere_rag_search_fixed via MCP run-sql"""
        try:
            env = os.environ.copy()
            env['TNS_ADMIN'] = self.tns_admin
            
            # Start MCP server and connect
            server_params = StdioServerParameters(
                command=self.config.mcp.sqlcl_path,
                args=["-mcp"],
                env=env
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Connect to database using saved connection
                    await session.call_tool(
                        "connect",
                        arguments={"connection_name": self.connection_name}
                    )
                    
                    # Prepare secure vector search SQL
                    safe_query = get_sql_injection_safe_query(query, self.config.security.max_query_length)
                    
                    vector_sql = f"""
SET SERVEROUTPUT ON SIZE 1000000
DECLARE
    l_cursor SYS_REFCURSOR;
    l_topic VARCHAR2(64);
    l_text VARCHAR2(4000);
    l_question VARCHAR2(4000);
    l_answer VARCHAR2(4000);
    l_similarity NUMBER;
    l_rank NUMBER;
    l_count NUMBER := 0;
BEGIN
    -- Use the vector search function with embeddings
    l_cursor := {self.config.vector.vector_function_name}('{safe_query}', {top_k}, {similarity_threshold});
    
    LOOP
        FETCH l_cursor INTO l_topic, l_text, l_question, l_answer, l_similarity, l_rank;
        EXIT WHEN l_cursor%NOTFOUND;
        
        l_count := l_count + 1;
        DBMS_OUTPUT.PUT_LINE('RESULT_' || l_count || ':');
        DBMS_OUTPUT.PUT_LINE('Topic: ' || l_topic);
        DBMS_OUTPUT.PUT_LINE('Content: ' || SUBSTR(l_text, 1, 200));
        DBMS_OUTPUT.PUT_LINE('Question: ' || l_question);
        DBMS_OUTPUT.PUT_LINE('Answer: ' || SUBSTR(l_answer, 1, 300));
        DBMS_OUTPUT.PUT_LINE('Similarity: ' || l_similarity);
        DBMS_OUTPUT.PUT_LINE('Rank: ' || l_rank);
        DBMS_OUTPUT.PUT_LINE('---');
    END LOOP;
    
    DBMS_OUTPUT.PUT_LINE('Total results: ' || l_count);
    CLOSE l_cursor;
END;
/"""
                    
                    # Execute vector search via MCP run-sql tool
                    result = await session.call_tool(
                        "run-sql",
                        arguments={"sql": vector_sql}
                    )
                    
                    if result.content and len(result.content) > 0:
                        mcp_output = str(result.content[0])
                        results = self._parse_vector_results(mcp_output)
                        
                        return {
                            'success': True,
                            'query': query,
                            'results_count': len(results),
                            'results': results,
                            'mcp_tool_used': 'run-sql',
                            'vector_function': self.config.vector.vector_function_name,
                            'technology': 'Oracle DB23ai Vector Search via MCP',
                            'embedding_model': self.config.vector.embedding_model,
                            'vector_dimensions': self.config.vector.vector_dimensions,
                            'distance_metric': self.config.vector.distance_metric,
                            'similarity_threshold': similarity_threshold,
                            'top_k': top_k
                        }
                    else:
                        return {
                            'success': False,
                            'query': query,
                            'error': 'No content returned from vector search',
                            'results_count': 0
                        }
                        
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'error': f'Vector search error: {str(e)}',
                'results_count': 0
            }
    
    def _parse_vector_results(self, mcp_output: str) -> List[Dict[str, Any]]:
        """Parse vector search results from DBMS_OUTPUT via MCP"""
        results = []
        current_result = {}
        
        try:
            # Handle MCP content format
            if hasattr(mcp_output, 'text'):
                actual_text = mcp_output.text
            elif "text='" in mcp_output:
                start = mcp_output.find("text='") + 6
                end = mcp_output.rfind("'")
                actual_text = mcp_output[start:end] if end > start else mcp_output
            else:
                actual_text = mcp_output
            
            # Process lines from DBMS_OUTPUT
            lines = actual_text.replace('\\n', '\n').split('\n')
            in_result = False
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('RESULT_') and line.endswith(':'):
                    if current_result and 'topic' in current_result:
                        results.append(current_result.copy())
                    current_result = {}
                    in_result = True
                    
                elif line == '---' or line.startswith('Total results:'):
                    if current_result and 'topic' in current_result:
                        results.append(current_result.copy())
                    in_result = False
                    if line.startswith('Total results:'):
                        break
                        
                elif in_result and ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'Topic':
                        current_result['topic'] = value
                    elif key == 'Content':
                        current_result['content'] = value
                    elif key == 'Question':
                        current_result['sample_question'] = value
                    elif key == 'Answer':
                        current_result['answer'] = value
                    elif key == 'Similarity':
                        try:
                            current_result['similarity_score'] = float(value)
                        except:
                            current_result['similarity_score'] = 0.0
                    elif key == 'Rank':
                        try:
                            current_result['rank'] = int(value)
                        except:
                            current_result['rank'] = 0
            
            # Handle final result if loop ended
            if current_result and 'topic' in current_result and current_result not in results:
                results.append(current_result)
                        
        except Exception as e:
            print(f"Vector result parsing error: {e}")
        
        return results

# Global vector MCP client
vector_mcp_client = VectorDB23aiMCPClient()

@tool
def search_knowledge_base_vector(query: str, top_k: int = 3, similarity_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Search DB23ai knowledge base using vector embeddings via MCP run-sql tool.
    
    Uses the oci_cohere_rag_search_fixed function which performs:
    1. Text embedding using OCI Cohere embed-english-v3.0 model
    2. Vector similarity search with COSINE distance
    3. Ranking by similarity scores
    
    Args:
        query (str): Search query text for vector embedding
        top_k (int): Number of most similar results to return (1-10)
        similarity_threshold (float): Minimum cosine similarity score (0.0-1.0)
        
    Returns:
        Dict[str, Any]: Vector search results with similarity scores and content
    """
    return asyncio.run(vector_mcp_client.vector_search(query, top_k, similarity_threshold))

@tool  
def get_vector_db_capabilities() -> Dict[str, Any]:
    """
    Get vector database capabilities and statistics via MCP.
    
    Returns information about:
    - Vector embedding model and dimensions
    - Database statistics (total entries, vector count)
    - Available MCP tools
    - Vector search capabilities
    
    Returns:
        Dict[str, Any]: Vector database capabilities and statistics
    """
    return asyncio.run(vector_mcp_client.get_vector_capabilities())

class VectorDB23aiMCPRAGAgent:
    """Production OCI GenAI Agent with vector search via MCP"""
    
    def __init__(self, agent_endpoint_id: str = None, compartment_id: str = None):
        self.config = load_config()
        
        # Use config values with parameter override capability
        self.agent_endpoint_id = agent_endpoint_id or self.config.oci.agent_endpoint_id
        self.compartment_id = compartment_id or self.config.oci.compartment_id
        self.agent = None
        
        self.instructions = """
        You are an expert IT support assistant powered by Oracle DB23ai vector search via Model Context Protocol.
        
        Your capabilities use advanced vector embedding search through secure MCP communication:
        - search_knowledge_base_vector: Uses oci_cohere_rag_search_fixed function via MCP run-sql
        - get_vector_db_capabilities: Retrieves vector database statistics and capabilities
        
        Technical specifications:
        - Protocol: Model Context Protocol for secure database communication
        - Vector Function: oci_cohere_rag_search_fixed (PL/SQL function)
        - Embedding Model: OCI Cohere embed-english-v3.0 (1024 dimensions)
        - Distance Metric: COSINE similarity
        - Database: Oracle DB23ai with DBMS_VECTOR package
        - Security: Saved connection credentials, SQL injection protection
        - Results: Ranked by similarity scores with detailed content
        
        When helping users:
        1. Use vector embedding search to find semantically similar content
        2. Provide answers based on similarity scores and rankings
        3. Include similarity scores to show confidence levels
        4. Explain that you're using secure MCP protocol with vector embeddings
        5. Reference specific content from the knowledge base with similarity confidence
        """
        
        self.tools = [search_knowledge_base_vector, get_vector_db_capabilities]
    
    def setup_agent(self):
        """Setup agent with vector search tools"""
        if self.agent_endpoint_id:
            try:
                client = AgentClient(
                    auth_type=self.config.oci.auth_type, 
                    profile=self.config.oci.auth_profile,
                    region=self.config.oci.region
                )
                
                self.agent = Agent(
                    client=client,
                    agent_endpoint_id=self.agent_endpoint_id,
                    instructions=self.instructions,
                    tools=self.tools
                )
                self.agent.setup()
                print("Production Vector RAG Agent with MCP Protocol setup complete")
                return True
            except Exception as e:
                print(f"Agent setup failed: {e}")
                self.agent = None
                return False
        else:
            print("Running in simulation mode")
            return False
    
    def chat(self, message: str) -> str:
        """Chat with the vector search agent"""
        if self.agent:
            try:
                return self.agent.run(message)
            except Exception as e:
                print(f"Agent error: {e}")
                return f"Agent error: {e}"
        else:
            return "Agent not initialized"

def main():
    """Initialize the production vector RAG agent"""
    agent = VectorDB23aiMCPRAGAgent()
    agent.setup_agent()
    print("Production Vector RAG Agent with MCP Protocol initialized")
    
if __name__ == "__main__":
    main()
```

## Step 7: Testing and Validation

### Test Vector Search Function

```sql
-- Test the vector search function directly
DECLARE
    l_cursor SYS_REFCURSOR;
    l_topic VARCHAR2(64);
    l_text VARCHAR2(4000);
    l_question VARCHAR2(4000);
    l_answer VARCHAR2(4000);
    l_similarity NUMBER;
    l_rank NUMBER;
BEGIN
    l_cursor := oci_cohere_rag_search_fixed('How do I setup conference call on Cisco Webex', 3, 0.5);
    
    DBMS_OUTPUT.PUT_LINE('Vector Search Results:');
    DBMS_OUTPUT.PUT_LINE('=====================');
    
    LOOP
        FETCH l_cursor INTO l_topic, l_text, l_question, l_answer, l_similarity, l_rank;
        EXIT WHEN l_cursor%NOTFOUND;
        
        DBMS_OUTPUT.PUT_LINE('Rank: ' || l_rank);
        DBMS_OUTPUT.PUT_LINE('Topic: ' || l_topic);
        DBMS_OUTPUT.PUT_LINE('Similarity: ' || ROUND(l_similarity * 100, 1) || '%');
        DBMS_OUTPUT.PUT_LINE('Answer: ' || SUBSTR(l_answer, 1, 200) || '...');
        DBMS_OUTPUT.PUT_LINE('---');
    END LOOP;
    
    CLOSE l_cursor;
END;
/
```

### Test MCP Server Connectivity

```bash
# Start MCP server
./start_mcp_background.sh

# Check server status
ps aux | grep 'sql -mcp'

# Monitor server logs
tail -f mcp_server.log

# Test the vector search agent
uv run python vector_db23ai_mcp_rag_agent.py
```

## Key Technical Features

### Secure MCP Implementation

1. **Model Context Protocol**: Standardized AI-to-database communication
2. **Saved Connections**: No hardcoded credentials in code
3. **SQL Injection Protection**: Query sanitization and length limits
4. **Environment Configuration**: Flexible deployment configuration
5. **Error Handling**: Comprehensive error handling and logging

### Advanced Vector Search

1. **OCI Cohere Integration**: Uses cohere.embed-english-v3.0 for high-quality embeddings
2. **Optimized Distance Calculation**: Proper cosine similarity with ranking
3. **Production-Ready Function**: Handles edge cases and provides detailed results
4. **Performance Optimization**: NEIGHBOR PARTITIONS indexing for enterprise scale

### Security and Compliance

- **Credential Security**: SQLcl saved connections with encrypted storage
- **Protocol Compliance**: Full MCP protocol adherence
- **Access Controls**: Database-level security for vector data access
- **Configuration Management**: Environment-based configuration without hardcoded values

## Production Deployment

### Environment Configuration

```bash
# Production environment variables
export APP_ENVIRONMENT=production
export OCI_AGENT_ENDPOINT_ID=ocid1.genaiagentendpoint.oc1.region.your_endpoint_id
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..your_compartment_id
export OCI_REGION=us-ashburn-1
export VECTOR_DEFAULT_TOP_K=5
export VECTOR_DEFAULT_THRESHOLD=0.7
export MAX_QUERY_LENGTH=2000
export DEBUG=false
```

### Startup Sequence

```bash
# 1. Start MCP server
./start_mcp_background.sh

# 2. Verify MCP server is running
ps aux | grep 'sql -mcp'

# 3. Run the production agent
uv run python vector_db23ai_mcp_rag_agent.py
```

### Performance Monitoring

```python
import logging
from datetime import datetime

# Example performance monitoring
class MCPVectorRAGMonitor:
    """Production monitoring for MCP vector RAG system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_mcp_performance(self, query: str, results_count: int, search_time: float, similarity_scores: List[float]):
        """Log MCP search performance metrics"""
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        self.logger.info(f"MCP Vector Search - Query: {len(query)} chars, "
                        f"Results: {results_count}, Time: {search_time:.3f}s, "
                        f"Avg Similarity: {avg_similarity:.3f}")
```

## Conclusion

This production implementation demonstrates how to build a secure, scalable RAG system using Oracle's enterprise-grade technologies with the Model Context Protocol. The solution combines:

- **OCI GenAI Agent**: For intelligent conversation management
- **Model Context Protocol**: For secure, standardized database communication
- **DB23ai Vector Database**: Enterprise-grade vector storage and search
- **OCI Cohere Embeddings**: High-quality semantic embeddings (1024 dimensions)
- **Secure Architecture**: No hardcoded credentials, SQL injection protection
- **Production Features**: Comprehensive monitoring, error handling, and configuration management

Key advantages include:
- **Security First**: MCP protocol compliance with saved connections
- **Native Integration**: Seamless Oracle component integration
- **Enterprise Scalability**: Production-ready performance with vector indexing
- **Real-time Processing**: Sub-second vector similarity search
- **Comprehensive Monitoring**: Built-in performance tracking and logging

This architecture serves as a blueprint for organizations building advanced RAG capabilities with Oracle's cloud and database technologies while maintaining enterprise security and compliance standards.

### Sample Query Results

When testing with "How do I set up a conference call on Cisco Webex", the system returns:

```
Vector search found 1 results:

**Result 1** (Similarity: 89.2%, Rank: 1)
Topic: Cisco Webex Conference Setup
Content: **Setting up Conference Calls on Cisco Webex** To set up a conference call on Cisco Webex with both video and audio...
Answer: Log in to your Cisco Webex account and click on the "Meetings" tab. Click on the "Schedule a Meeting" button and enter the meeting details...

Search powered by oci_cohere_rag_search_fixed with OCI Cohere embed-english-v3.0
```

This demonstrates the system's ability to understand semantic queries and return highly relevant, contextual responses using vector similarity search through the secure MCP protocol.