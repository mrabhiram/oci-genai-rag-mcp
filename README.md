# Building a RAG System with OCI GenAI Agent, DB23ai Vector Search, and Model Context Protocol

## Introduction

This comprehensive guide demonstrates how to build a Retrieval-Augmented Generation (RAG) system using Oracle Cloud Infrastructure (OCI) GenAI Agent integrated with Oracle Database 23ai's native vector capabilities through the Model Context Protocol (MCP). The solution leverages DB23ai's built-in DBMS_VECTOR package, OCI's Cohere embedding models, and MCP for secure, standardized database communication.

Unlike traditional approaches that require separate vector databases or direct SQL execution, this solution utilizes DB23ai's enterprise-grade vector capabilities with MCP protocol compliance for secure, scalable database interactions.

## Architecture Overview

The solution follows a secure, protocol-compliant architecture:
![OCI_Genai_RAG_DB23ai](https://github.com/user-attachments/assets/79a80150-04ea-41ba-b465-28186a56396c)


Key components:
- **OCI GenAI Agent**: Orchestrates conversations and function calls
- **MCP Protocol**: Standardized communication between AI and database
- **DB23ai MCP Server**: Native MCP server for secure database access
- **DB23ai Vector Store**: Native vector storage with DBMS_VECTOR capabilities
- **OCI Cohere Embeddings**: Uses cohere.embed-english-v3.0 model (1024 dimensions)
- **Vector Search Function**: Optimized `oci_cohere_rag_search_fixed` for similarity search
- **Interactive Demo**: An Interactive demo to test the agent

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

### Create Saved Database Connection( or run start_mcp_background.sh)

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



## Step 3: Database Setup and Data Loading

### Configure OCI Credentials for DBMS_VECTOR

Oracle provides mechanisms like DBMS_VECTOR.CREATE_CREDENTIAL to securely store authentication details (like usernames, passwords, or tokens) within the database itself, rather than embedding them directly in your code. This helps maintain security and avoids exposing sensitive information.Set up credentials for OCI GenAI service integration:

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

https://www.kaggle.com/datasets/dkhundley/sample-rag-knowledge-item-dataset?resource=download

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

Generate embeddings in-DB with VECTOR_EMBEDDING (OCI Cohere) to centralize governance, eliminate ETL, and maintain query–document model parity over an MCP-only path. Add a COSINE vector index  to convert KNN into low-latency ANN, improving recall and TTFT at scale while keeping the ADK Function Tool a thin orchestrator. Net result: stronger grounding, fewer hallucinations, and predictable throughput.

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

## Step 4: Create RAG Search Function
This PL/SQL function implements semantic vector search using Oracle's DBMS_VECTOR package with OCI Cohere embeddings for RAG
  (Retrieval-Augmented Generation) operations
###  Key Technical Features

  1. Real-time Query Embedding
  2. Semantic Similarity Calculation
  3. Intelligent Filtering & Ranking
  4. High Quality Relevance Scoring
  5. Multilingual & Synonym Support

This is a significant upgrade from simple keyword search, providing intelligent, context-aware RAG capabilities that understand
  user intent rather than just matching text strings
Create a RAG search function with proper vector distance calculations:

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

### Configure MCP Server Startup Script

Create a secure startup script for the DB23ai MCP server:
This shell script creates an sql -mcp server and saves the connection named mcp_saved which will be used by the python agent to connect.

[start_mcp_background.sh](start_mcp_background.sh)
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

### Secure Configuration Module
Storing Configuration module for MCP, Database, OCI GenAI Agent,  Vector Search, Tuning and Application Configuration. Below are the environment variables used for config.py with defaults. Changes are necessary to your environment.

[config.py](config.py)

```python

"""
Copyright (c) 2024, 2025, Oracle and/or its affiliates.
Licensed under the Universal Permissive License v1.0 as shown at http://oss.oracle.com/licenses/upl.

Configuration management for Vector DB23ai MCP RAG Agent
Supports environment variables and secure credential management

REQUIRED ENVIRONMENT VARIABLES:
==============================================

# Essential MCP and Database Configuration:
export TNS_ADMIN=/home/ubuntu/wallet-db                    # Path to Oracle wallet directory
export MCP_SQLCL_PATH=/home/ubuntu/adk-rag-mcp/sqlcl/bin/sql  # Path to SQLcl binary with MCP support
export MCP_CONNECTION_NAME=mcp_saved                       # Name of saved SQLcl connection (created via: connect -save mcp_saved)

# OCI GenAI Agent Configuration (Replace with your actual values):
export OCI_AGENT_ENDPOINT_ID=ocid1.genaiagentendpoint.oc1.region.your_endpoint_id  # Your OCI GenAI Agent endpoint OCID
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..your_compartment_id              # Your OCI compartment OCID
export OCI_REGION=us-ashburn-1                             # Your OCI region (e.g., us-ashburn-1, us-phoenix-1)

# OPTIONAL - Vector Search Tuning (Uses defaults if not set):
export VECTOR_FUNCTION_NAME=oci_cohere_rag_search_fixed    # Database vector search function name
export VECTOR_DEFAULT_TOP_K=3                              # Default number of results to return
export VECTOR_DEFAULT_THRESHOLD=0.6                        # Default similarity threshold (0.0-1.0)
export MAX_QUERY_LENGTH=1000                               # Maximum query length for security

# OPTIONAL - Application Configuration:
export APP_ENVIRONMENT=production                          # Set to 'production' for production deployment
export DEBUG=false                                         # Set to 'true' for debug logging
export OCI_AUTH_TYPE=api_key                              # OCI authentication type
export OCI_AUTH_PROFILE=DEFAULT                           # OCI CLI profile name

PREREQUISITES:
==============
1. Oracle DB23ai database with vector embeddings setup
2. SQLcl installed with MCP support (Java 17+ required)
3. Database wallet configured and accessible
4. Saved connection created in SQLcl: connect -save mcp_saved USERNAME/"PASSWORD"@DSN
5. OCI CLI configured with proper authentication
6. MCP server started: ./start_mcp_background.sh

QUICK START EXAMPLE:
===================
export TNS_ADMIN=/home/ubuntu/wallet-db
export MCP_CONNECTION_NAME=mcp_saved
export OCI_AGENT_ENDPOINT_ID=ocid1.genaiagentendpoint.oc1.iad.amaaaaaaieosruaa...
export OCI_REGION=us-ashburn-1
uv run python vector_db23ai_mcp_rag_agent.py


"""
```
./config.py

## Step 6: Implement MCP-Based Vector Search Agent

### MCP RAG Agent
This is the main code that contains RAG functions exposed via custom tools. These are the two functions exposed via the rag_agent. 

[db23ai_mcp_rag_agent.py](db23ai_mcp_rag_agent.py)

**search_knowledge_base_vector**(query, top_k=3, similarity_threshold=0.6) — Exposes MCP-only retrieval. The tool launches SQLcl over MCP, connects via a saved DB23ai credential, and invokes your PL/SQL entry point (oci_cohere_rag_search_fixed) to embed the query (Cohere embed-english-v3.0, COSINE) and run a KNN/ANN search. It parses DBMS_OUTPUT into ranked results with topic, snippet, similarity, and rank, returning a clean JSON payload the agent can ground on.

**get_vector_db_capabilities**() — Surfaces vector “health and config” to the agent. Via MCP it should report the active embedding model and dimensions, corpus/vector counts, available MCP tools, and supported distance/search modes—useful for readiness checks, guardrails, and dynamic tool selection. 

**Sample Output:**

```python
ubuntu@ubuntu-host:~/oci-genai-rag-mcp$ uv run python vector_db23ai_mcp_rag_agent.py 
[08/11/25 20:21:37]  INFO     Checking integrity of agent details...                                                                    
                     INFO     Checking synchronization of local and remote agent settings...                                            
                     INFO     Checking synchronization of local and remote function tools...                                            
╭──────────── Local and remote function tools found ─────────────╮
│ Local function tools (2):                                      │
│ ['get_vector_db_capabilities', 'search_knowledge_base_vector'] │
│                                                                │
│ Remote function tools (0):                                     │
│ []                                                             │
╰────────────────────────────────────────────────────────────────╯
[08/11/25 20:21:38]  INFO     Adding local tool search_knowledge_base_vector to remote...                                               
[08/11/25 20:21:43]  INFO     Waiting for tool ocid1.genaiagenttool.oc1.iad.amaaaaaaieosruaaaxkpx55cy5kahxm4s6viyoyfrsmypms3bfosor3aaefa
                              to be active...                                                                                           
[08/11/25 20:21:48]  INFO     Waiting for tool ocid1.genaiagenttool.oc1.iad.amaaaaaaieosruaaaxkpx55cy5kahxm4s6viyoyfrsmypms3bfosor3aaefa
                              to be active...                                                                                           
[08/11/25 20:21:53]  INFO     Waiting for tool ocid1.genaiagenttool.oc1.iad.amaaaaaaieosruaaaxkpx55cy5kahxm4s6viyoyfrsmypms3bfosor3aaefa
                              to be active...                                                                                           
                     INFO     Adding local tool get_vector_db_capabilities to remote...                                                 
[08/11/25 20:21:59]  INFO     Waiting for tool ocid1.genaiagenttool.oc1.iad.amaaaaaaieosruaaa26ax76njbkfvep7r5chxfvkymiyvvyrn3xqwqnk2iva
                              to be active...                                                                                           
[08/11/25 20:22:04]  INFO     Waiting for tool ocid1.genaiagenttool.oc1.iad.amaaaaaaieosruaaa26ax76njbkfvep7r5chxfvkymiyvvyrn3xqwqnk2iva
                              to be active...                                                                                           
[08/11/25 20:22:09]  INFO     Waiting for tool ocid1.genaiagenttool.oc1.iad.amaaaaaaieosruaaa26ax76njbkfvep7r5chxfvkymiyvvyrn3xqwqnk2iva
                              to be active...                                                                                           
                     INFO     Checking synchronization of local and remote RAG tools...                                                 
                     INFO     Checking synchronization of local and remote SQL tools...                                                 
╭─ Local and remote SQL tools found ─╮
│ Local SQL tools (0):               │
│ []                                 │
│                                    │
│ Remote SQL tools (0):              │
│ []                                 │
╰────────────────────────────────────╯
OCI GenAI Agent with DB23ai Vector Search setup complete
Vector DB23ai MCP RAG Agent initialized
```

## Step 7: Testing and Validation




### Project Startup Sequence

```bash
# Start MCP server
./start_mcp_background.sh

# Check server status
ps aux | grep 'sql -mcp'

# Monitor server logs
tail -f mcp_server.log

# Make sure to checkout the environment variables in config.py and are satisfied.
export OCI_AGENT_ENDPOINT_ID=ocid1.genaiagentendpoint.oc1.region.your_endpoint_id
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..your_compartment_id
export OCI_REGION=us-ashburn-1
export TNS_ADMIN=/path/to/db/wallet
export MCP_CONNECTION_NAME=mcp_saved # Name of saved SQLcl connection (created via start_mcp_background.sh)
export MCP_SQLCL_PATH=/path/to/sqlcl/bin/sql
export VECTOR_DEFAULT_TOP_K=5
export VECTOR_DEFAULT_THRESHOLD=0.7
export MAX_QUERY_LENGTH=2000
export DEBUG=false

# Run the Vector Search agent
uv run python vector_db23ai_mcp_rag_agent.py

# Test the agent with an interactice ADK client
uv run python simple_demo.py

Oracle GenAI + DB23ai Vector Search Demo
==================================================
Type 'quit' to exit

You: how to setup conference call?
╭───────────────────────────── Chat request to remote agent: None ─────────────────────────────╮
│ (Local --> Remote)                                                                           │
│                                                                                              │
│ user message:                                                                                │
│ how to setup conference call?                                                                │
│                                                                                              │
│ performed actions by client:                                                                 │
│ []                                                                                           │
│                                                                                              │
│ session id:                                                                                  │
│ ocid1.genaiagentsession.oc1.iad.amaaaaaa7mjirbaa7srd6yn7ph5dialttb3kdggtyls4y3kz2ehgi4jx2una │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────── Chat response from remote agent ────────────────────────────────────────────╮
│ (Local <-- Remote)                                                                                                     │
│                                                                                                                        │
│ agent message:                                                                                                         │
│ null                                                                                                                   │
│                                                                                                                        │
│ required actions for client to take:                                                                                   │
│ [                                                                                                                      │
│     {                                                                                                                  │
│         "action_id": "418aa5f8-468a-4d79-bd57-fd80bb858484",                                                           │
│         "required_action_type": "FUNCTION_CALLING_REQUIRED_ACTION",                                                    │
│         "function_call": {                                                                                             │
│             "name": "search_knowledge_base_vector",                                                                    │
│             "arguments": "{\"query\": \"setup conference call\", \"top_k\": \"5\", \"similarity_threshold\": \"0.5\"}" │
│         }                                                                                                              │
│     }                                                                                                                  │
│ ]                                                                                                                      │
│                                                                                                                        │
│ guardrail result:                                                                                                      │
│ None                                                                                                                   │
│                                                                                                                        │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────── Function call requested by agent and mapped local handler function ───────╮
│ Agent function tool name:                                                       │
│ search_knowledge_base_vector                                                    │
│                                                                                 │
│ Agent function tool call arguments:                                             │
│ {'query': 'setup conference call', 'top_k': '5', 'similarity_threshold': '0.5'} │
│                                                                                 │
│ Mapped local handler function name:                                             │
│ search_knowledge_base_vector                                                    │
╰─────────────────────────────────────────────────────────────────────────────────╯
---------- MCP SERVER STARTUP ----------
MCP Server started successfully on Mon Aug 18 21:31:57 UTC 2025
Press Ctrl+C to stop the server
----------------------------------------
Aug 18, 2025 9:31:57 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
INFO: Client initialize request - Protocol: 2025-06-18, Capabilities: ClientCapabilities[experimental=null, roots=null, sampling=null], Info: Implementation[name=mcp, version=0.1.0]
Aug 18, 2025 9:31:57 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
WARNING: Client requested unsupported protocol version: 2025-06-18, so the server will sugggest the 2024-11-05 version instead
╭─────────────────────────────────────────────────── Obtained local function execution result ────────────────────────────────────────────────────╮
│ {'success': True, 'query': 'setup conference call', 'results_count': 2, 'results': [{'topic': 'Setting Up a Conference Call on Cisco Webex',    │
│ 'content': 'To set up a conference call on Cisco Webex, follow these steps:', 'sample_question': 'How do I set up a conference call on Cisco    │
│ Webex with both video and audio for a meeting with multiple participants?', 'answer': 'To set up a conference call on Cisco Webex with both     │
│ video and audio for a meeting with multiple participants, follow these steps:', 'similarity_score': 0.565289, 'rank': 1}, {'topic': 'Setting Up │
│ a Conference Call on Cisco Webex', 'content': 'To set up a conference call on Cisco Webex, follow these steps:', 'sample_question': 'How do I   │
│ set up a conference call on Cisco Webex with both video and audio for a meeting with multiple participants?', 'answer': 'To set up a conference │
│ call on Cisco Webex with both video and audio for a meeting with multiple participants, follow these steps:', 'similarity_score': 0.565289,     │
│ 'rank': 1}], 'mcp_tool_used': 'run-sql', 'vector_function': 'oci_cohere_rag_search_fixed', 'technology': 'Oracle DB23ai Vector Search via MCP', │
│ 'embedding_model': 'cohere.embed-english-v3.0', 'vector_dimensions': 1024, 'distance_metric': 'COSINE', 'similarity_threshold': 0.5, 'top_k':   │
│ 5}                                                                                                                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────── Chat request to remote agent: None ───────────────────────────────────────────────────────╮
│ (Local --> Remote)                                                                                                                              │
│                                                                                                                                                 │
│ user message:                                                                                                                                   │
│ null                                                                                                                                            │
│                                                                                                                                                 │
│ performed actions by client:                                                                                                                    │
│ [{'action_id': '418aa5f8-468a-4d79-bd57-fd80bb858484', 'performed_action_type': 'FUNCTION_CALLING_PERFORMED_ACTION', 'function_call_output':    │
│ '{"success": true, "query": "setup conference call", "results_count": 2, "results": [{"topic": "Setting Up a Conference Call on Cisco Webex",   │
│ "content": "To set up a conference call on Cisco Webex, follow these steps:", "sample_question": "How do I set up a conference call on Cisco    │
│ Webex with both video and audio for a meeting with multiple participants?", "answer": "To set up a conference call on Cisco Webex with both     │
│ video and audio for a meeting with multiple participants, follow these steps:", "similarity_score": 0.565289, "rank": 1}, {"topic": "Setting Up │
│ a Conference Call on Cisco Webex", "content": "To set up a conference call on Cisco Webex, follow these steps:", "sample_question": "How do I   │
│ set up a conference call on Cisco Webex with both video and audio for a meeting with multiple participants?", "answer": "To set up a conference │
│ call on Cisco Webex with both video and audio for a meeting with multiple participants, follow these steps:", "similarity_score": 0.565289,     │
│ "rank": 1}], "mcp_tool_used": "run-sql", "vector_function": "oci_cohere_rag_search_fixed", "technology": "Oracle DB23ai Vector Search via MCP", │
│ "embedding_model": "cohere.embed-english-v3.0", "vector_dimensions": 1024, "distance_metric": "COSINE", "similarity_threshold": 0.5, "top_k":   │
│ 5}'}]                                                                                                                                           │
│                                                                                                                                                 │
│ session id:                                                                                                                                     │
│ ocid1.genaiagentsession.oc1.iad.amaaaaaa7mjirbaa7srd6yn7ph5dialttb3kdggtyls4y3kz2ehgi4jx2una                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────── Chat response from remote agent ────────────────────────────────────────────────────────╮
│ (Local <-- Remote)                                                                                                                              │
│                                                                                                                                                 │
│ agent message:                                                                                                                                  │
│ {                                                                                                                                               │
│     "role": "AGENT",                                                                                                                            │
│     "content": {                                                                                                                                │
│         "text": "To set up a conference call on Cisco Webex with both video and audio for a meeting with multiple participants, follow these    │
│ steps. The similarity score for this result is 0.565289, indicating a moderate level of confidence in the answer. \n\nPlease note that the      │
│ provided information is based on the search results from the knowledge base, and it's recommended to consult the official Cisco Webex           │
│ documentation or support resources for the most up-to-date and accurate instructions. \n\nThe search was performed using the Oracle DB23ai      │
│ vector search function, oci_cohere_rag_search_fixed, with the cohere.embed-english-v3.0 embedding model and COSINE distance metric. The top 5   │
│ results were returned, with a similarity threshold of 0.5. \n\nFor more information or specific guidance on setting up a conference call, you   │
│ can refer to the official Cisco Webex support resources or consult with a qualified IT professional.",                                          │
│         "citations": null,                                                                                                                      │
│         "paragraph_citations": null                                                                                                             │
│     },                                                                                                                                          │
│     "time_created": "2025-08-18T21:32:08.194000+00:00"                                                                                          │
│ }                                                                                                                                               │
│                                                                                                                                                 │
│ required actions for client to take:                                                                                                            │
│ null                                                                                                                                            │
│                                                                                                                                                 │
│ guardrail result:                                                                                                                               │
│ None                                                                                                                                            │
│                                                                                                                                                 │
│                                                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Error: 'RunResponse' object has no attribute 'get'


You: What steps can I take to fix my printer when it's jammed and won't print?
╭───────────────────────────── Chat request to remote agent: None ─────────────────────────────╮
│ (Local --> Remote)                                                                           │
│                                                                                              │
│ user message:                                                                                │
│ What steps can I take to fix my printer when it's jammed and won't print?                    │
│                                                                                              │
│ performed actions by client:                                                                 │
│ []                                                                                           │
│                                                                                              │
│ session id:                                                                                  │
│ ocid1.genaiagentsession.oc1.iad.amaaaaaa7mjirbaans7zfhmtnhilezaxg3irqx3ne6gcmdt7ad56hhyqqoia │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────── Chat response from remote agent ─────────────────────────────────────────────╮
│ (Local <-- Remote)                                                                                                       │
│                                                                                                                          │
│ agent message:                                                                                                           │
│ null                                                                                                                     │
│                                                                                                                          │
│ required actions for client to take:                                                                                     │
│ [                                                                                                                        │
│     {                                                                                                                    │
│         "action_id": "c4713348-85ff-48db-8e4f-83dc685bc7ce",                                                             │
│         "required_action_type": "FUNCTION_CALLING_REQUIRED_ACTION",                                                      │
│         "function_call": {                                                                                               │
│             "name": "search_knowledge_base_vector",                                                                      │
│             "arguments": "{\"query\": \"fixing a jammed printer\", \"top_k\": \"5\", \"similarity_threshold\": \"0.5\"}" │
│         }                                                                                                                │
│     }                                                                                                                    │
│ ]                                                                                                                        │
│                                                                                                                          │
│ guardrail result:                                                                                                        │
│ None                                                                                                                     │
│                                                                                                                          │
│                                                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────── Function call requested by agent and mapped local handler function ────────╮
│ Agent function tool name:                                                         │
│ search_knowledge_base_vector                                                      │
│                                                                                   │
│ Agent function tool call arguments:                                               │
│ {'query': 'fixing a jammed printer', 'top_k': '5', 'similarity_threshold': '0.5'} │
│                                                                                   │
│ Mapped local handler function name:                                               │
│ search_knowledge_base_vector                                                      │
╰───────────────────────────────────────────────────────────────────────────────────╯
---------- MCP SERVER STARTUP ----------
MCP Server started successfully on Mon Aug 18 21:37:02 UTC 2025
Press Ctrl+C to stop the server
----------------------------------------
Aug 18, 2025 9:37:03 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
INFO: Client initialize request - Protocol: 2025-06-18, Capabilities: ClientCapabilities[experimental=null, roots=null, sampling=null], Info: Implementation[name=mcp, version=0.1.0]
Aug 18, 2025 9:37:03 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
WARNING: Client requested unsupported protocol version: 2025-06-18, so the server will sugggest the 2024-11-05 version instead
╭─────────────────────────────────────────────────── Obtained local function execution result ────────────────────────────────────────────────────╮
│ {'success': True, 'query': 'fixing a jammed printer', 'results_count': 0, 'results': [], 'mcp_tool_used': 'run-sql', 'vector_function':         │
│ 'oci_cohere_rag_search_fixed', 'technology': 'Oracle DB23ai Vector Search via MCP', 'embedding_model': 'cohere.embed-english-v3.0',             │
│ 'vector_dimensions': 1024, 'distance_metric': 'COSINE', 'similarity_threshold': 0.5, 'top_k': 5}                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────── Chat request to remote agent: None ───────────────────────────────────────────────────────╮
│ (Local --> Remote)                                                                                                                              │
│                                                                                                                                                 │
│ user message:                                                                                                                                   │
│ null                                                                                                                                            │
│                                                                                                                                                 │
│ performed actions by client:                                                                                                                    │
│ [{'action_id': 'c4713348-85ff-48db-8e4f-83dc685bc7ce', 'performed_action_type': 'FUNCTION_CALLING_PERFORMED_ACTION', 'function_call_output':    │
│ '{"success": true, "query": "fixing a jammed printer", "results_count": 0, "results": [], "mcp_tool_used": "run-sql", "vector_function":        │
│ "oci_cohere_rag_search_fixed", "technology": "Oracle DB23ai Vector Search via MCP", "embedding_model": "cohere.embed-english-v3.0",             │
│ "vector_dimensions": 1024, "distance_metric": "COSINE", "similarity_threshold": 0.5, "top_k": 5}'}]                                             │
│                                                                                                                                                 │
│ session id:                                                                                                                                     │
│ ocid1.genaiagentsession.oc1.iad.amaaaaaa7mjirbaans7zfhmtnhilezaxg3irqx3ne6gcmdt7ad56hhyqqoia                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────── Chat response from remote agent ────────────────────────────────────────────────────────╮
│ (Local <-- Remote)                                                                                                                              │
│                                                                                                                                                 │
│ agent message:                                                                                                                                  │
│ {                                                                                                                                               │
│     "role": "AGENT",                                                                                                                            │
│     "content": {                                                                                                                                │
│         "text": "To fix a jammed printer, you can try the following steps:\n\n1. Turn off the printer and unplug it from the power source.\n2.  │
│ Locate the jammed paper and gently pull it out.\n3. Check for any remaining paper fragments or debris and remove them.\n4. Plug in the printer  │
│ and turn it back on.\n5. Try printing a test page to see if the issue is resolved.\n\nIf the problem persists, you may need to consult the      │
│ printer's user manual or contact the manufacturer's support for further assistance. \n\nNote: The provided information is based on general      │
│ knowledge and may not be specific to your particular printer model. It's always a good idea to consult the user manual or contact the           │
│ manufacturer's support for specific instructions on how to fix a jammed printer.",                                                              │
│         "citations": null,                                                                                                                      │
│         "paragraph_citations": null                                                                                                             │
│     },                                                                                                                                          │
│     "time_created": "2025-08-18T21:37:12.504000+00:00"                                                                                          │
│ }                                                                                                                                               │
│                                                                                                                                                 │
│ required actions for client to take:                                                                                                            │
│ null                                                                                                                                            │
│                                                                                                                                                 │
│ guardrail result:                                                                                                                               │
│ None                                                                                                                                            │
│                                                                                                                                                 │
│                                                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Error: 'RunResponse' object has no attribute 'get'


You: what are your capabilities?
╭───────────────────────────── Chat request to remote agent: None ─────────────────────────────╮
│ (Local --> Remote)                                                                           │
│                                                                                              │
│ user message:                                                                                │
│ what are your capabilities?                                                                  │
│                                                                                              │
│ performed actions by client:                                                                 │
│ []                                                                                           │
│                                                                                              │
│ session id:                                                                                  │
│ ocid1.genaiagentsession.oc1.iad.amaaaaaa7mjirbaajpagzxuybcfn33xcfcljlr5rv4g6cfll5pruybbq4bwq │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────── Chat response from remote agent ──────────────────╮
│ (Local <-- Remote)                                                  │
│                                                                     │
│ agent message:                                                      │
│ null                                                                │
│                                                                     │
│ required actions for client to take:                                │
│ [                                                                   │
│     {                                                               │
│         "action_id": "916346aa-688c-4b61-8e17-50a2ba6b2ec3",        │
│         "required_action_type": "FUNCTION_CALLING_REQUIRED_ACTION", │
│         "function_call": {                                          │
│             "name": "get_vector_db_capabilities",                   │
│             "arguments": "{}"                                       │
│         }                                                           │
│     }                                                               │
│ ]                                                                   │
│                                                                     │
│ guardrail result:                                                   │
│ None                                                                │
│                                                                     │
│                                                                     │
╰─────────────────────────────────────────────────────────────────────╯
╭─ Function call requested by agent and mapped local handler function ─╮
│ Agent function tool name:                                            │
│ get_vector_db_capabilities                                           │
│                                                                      │
│ Agent function tool call arguments:                                  │
│ {}                                                                   │
│                                                                      │
│ Mapped local handler function name:                                  │
│ get_vector_db_capabilities                                           │
╰──────────────────────────────────────────────────────────────────────╯
---------- MCP SERVER STARTUP ----------
MCP Server started successfully on Mon Aug 18 21:42:06 UTC 2025
Press Ctrl+C to stop the server
----------------------------------------
Aug 18, 2025 9:42:06 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
INFO: Client initialize request - Protocol: 2025-06-18, Capabilities: ClientCapabilities[experimental=null, roots=null, sampling=null], Info: Implementation[name=mcp, version=0.1.0]
Aug 18, 2025 9:42:06 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
WARNING: Client requested unsupported protocol version: 2025-06-18, so the server will sugggest the 2024-11-05 version instead
╭─────────────────────────────────────────────────── Obtained local function execution result ────────────────────────────────────────────────────╮
│ {'success': True, 'server_type': 'Oracle DB23ai Vector Search via MCP', 'vector_function': 'oci_cohere_rag_search_fixed', 'mcp_tool':           │
│ 'run-sql', 'capabilities': {'tools_count': 5, 'vector_search': True, 'embedding_generation': True, 'cosine_similarity': True},                  │
│ 'vector_statistics': {'Total_entries': '10', 'Vector_entries': '10', 'Embedding_model': 'cohere.embed-english-v3.0', 'Vector_dimensions':       │
│ '1024', 'Distance_metric': 'COSINE', 'Vector_function': 'oci_cohere_rag_search_fixed'}, 'tools': [{'name': 'list-connections', 'description':   │
│ 'List all available oracle named/saved connections in the connections storage\n\nThe `model` argument should specify only the name and version  │
│ of the LLM (Large Language Model) you are using, with no additional information.\nThe `mcp_client` argument should specify only the name of the │
│ MCP (Model Context Protocol) client you are using, with no additional information.\n'}, {'name': 'connect', 'description': 'Provides an         │
│ interface to connect to a specified database. If a database connection is already active, prompt the user for confirmation before switching to  │
│ the new connection. If no connection exists, list the available schemas for selection.\nthe connection name is case sensitive\nNote: If the     │
│ provided connection is invalid or does not match any saved connection, display instructions to the user on how to create a named connection in  │
│ SQLcl\n\n\nThe `model` argument should specify only the name and version of the LLM (Large Language Model) you are using, with no additional    │
│ information.\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with no          │
│ additional information.\n'}, {'name': 'disconnect', 'description': 'This tool performs a disconnection from the current session in an Oracle    │
│ database. If a user is connected, it logs out cleanly and returns to the SQL prompt without an active database connection.\n\nThe `model`       │
│ argument should specify only the name and version of the LLM (Large Language Model) you are using, with no additional information.\nThe         │
│ `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with no additional                 │
│ information.\n'}, {'name': 'run-sqlcl', 'description': 'This tool executes SQLcl commands in the SQLcl CLI. If the given command requires a     │
│ database connection, it prompts the user to connect using the connect tool.\nYou should:\n\n\tExecute the provided SQLcl command.\n\n\tReturn   │
│ the results.\n\nArgs:\n\n\tsql: The SQLcl command to execute.\n\nReturns:\n\n\tCommand results.\n\n\nThe `model` argument should specify only   │
│ the name and version of the LLM (Large Language Model) you are using, with no additional information.\nThe `mcp_client` argument should specify │
│ only the name of the MCP (Model Context Protocol) client you are using, with no additional information.\n'}, {'name': 'run-sql', 'description': │
│ 'This tool executes SQL queries in an Oracle database. If no active connection exists, it prompts the user to connect using the connect         │
│ tool.\n\nYou should:\n\n\tExecute the provided SQL query.\n\n\tReturn the results in CSV format.\n\nArgs:\n\n\tsql: The SQL query to            │
│ execute.\n\n\nThe `model` argument should specify only the name and version of the LLM (Large Language Model) you are using, with no additional │
│ information.\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with no          │
│ additional information.\n\nReturns:\n\n\tCSV-formatted query results.\nFor every SQL query you generate, please include a comment at the        │
│ beginning of the SELECT statement (or other main SQL command) that identifies the LLM model name and version you are using. Format the comment  │
│ as: /* LLM in use is  */ and place it immediately after the main SQL keyword.\nFor example:\n\nSELECT /* LLM in use is claude-sonnet-4 */       │
│ column1, column2 FROM table_name;\nINSERT /* LLM in use is claude-sonnet-4 */ INTO table_name VALUES (...);\nUPDATE /* LLM in use is            │
│ claude-sonnet-4 */ table_name SET ...;\n\nPlease apply this format consistently to all SQL queries you generate, using your actual model name   │
│ and version in the comment\n'}]}                                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────── Chat request to remote agent: None ───────────────────────────────────────────────────────╮
│ (Local --> Remote)                                                                                                                              │
│                                                                                                                                                 │
│ user message:                                                                                                                                   │
│ null                                                                                                                                            │
│                                                                                                                                                 │
│ performed actions by client:                                                                                                                    │
│ [{'action_id': '916346aa-688c-4b61-8e17-50a2ba6b2ec3', 'performed_action_type': 'FUNCTION_CALLING_PERFORMED_ACTION', 'function_call_output':    │
│ '{"success": true, "server_type": "Oracle DB23ai Vector Search via MCP", "vector_function": "oci_cohere_rag_search_fixed", "mcp_tool":          │
│ "run-sql", "capabilities": {"tools_count": 5, "vector_search": true, "embedding_generation": true, "cosine_similarity": true},                  │
│ "vector_statistics": {"Total_entries": "10", "Vector_entries": "10", "Embedding_model": "cohere.embed-english-v3.0", "Vector_dimensions":       │
│ "1024", "Distance_metric": "COSINE", "Vector_function": "oci_cohere_rag_search_fixed"}, "tools": [{"name": "list-connections", "description":   │
│ "List all available oracle named/saved connections in the connections storage\\n\\nThe `model` argument should specify only the name and        │
│ version of the LLM (Large Language Model) you are using, with no additional information.\\nThe `mcp_client` argument should specify only the    │
│ name of the MCP (Model Context Protocol) client you are using, with no additional information.\\n"}, {"name": "connect", "description":         │
│ "Provides an interface to connect to a specified database. If a database connection is already active, prompt the user for confirmation before  │
│ switching to the new connection. If no connection exists, list the available schemas for selection.\\nthe connection name is case               │
│ sensitive\\nNote: If the provided connection is invalid or does not match any saved connection, display instructions to the user on how to      │
│ create a named connection in SQLcl\\n\\n\\nThe `model` argument should specify only the name and version of the LLM (Large Language Model) you  │
│ are using, with no additional information.\\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client  │
│ you are using, with no additional information.\\n"}, {"name": "disconnect", "description": "This tool performs a disconnection from the current │
│ session in an Oracle database. If a user is connected, it logs out cleanly and returns to the SQL prompt without an active database             │
│ connection.\\n\\nThe `model` argument should specify only the name and version of the LLM (Large Language Model) you are using, with no         │
│ additional information.\\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with │
│ no additional information.\\n"}, {"name": "run-sqlcl", "description": "This tool executes SQLcl commands in the SQLcl CLI. If the given command │
│ requires a database connection, it prompts the user to connect using the connect tool.\\nYou should:\\n\\n\\tExecute the provided SQLcl         │
│ command.\\n\\n\\tReturn the results.\\n\\nArgs:\\n\\n\\tsql: The SQLcl command to execute.\\n\\nReturns:\\n\\n\\tCommand results.\\n\\n\\nThe   │
│ `model` argument should specify only the name and version of the LLM (Large Language Model) you are using, with no additional                   │
│ information.\\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with no         │
│ additional information.\\n"}, {"name": "run-sql", "description": "This tool executes SQL queries in an Oracle database. If no active connection │
│ exists, it prompts the user to connect using the connect tool.\\n\\nYou should:\\n\\n\\tExecute the provided SQL query.\\n\\n\\tReturn the      │
│ results in CSV format.\\n\\nArgs:\\n\\n\\tsql: The SQL query to execute.\\n\\n\\nThe `model` argument should specify only the name and version  │
│ of the LLM (Large Language Model) you are using, with no additional information.\\nThe `mcp_client` argument should specify only the name of    │
│ the MCP (Model Context Protocol) client you are using, with no additional information.\\n\\nReturns:\\n\\n\\tCSV-formatted query results.\\nFor │
│ every SQL query you generate, please include a comment at the beginning of the SELECT statement (or other main SQL command) that identifies the │
│ LLM model name and version you are using. Format the comment as: /* LLM in use is  */ and place it immediately after the main SQL               │
│ keyword.\\nFor example:\\n\\nSELECT /* LLM in use is claude-sonnet-4 */ column1, column2 FROM table_name;\\nINSERT /* LLM in use is             │
│ claude-sonnet-4 */ INTO table_name VALUES (...);\\nUPDATE /* LLM in use is claude-sonnet-4 */ table_name SET ...;\\n\\nPlease apply this format │
│ consistently to all SQL queries you generate, using your actual model name and version in the comment\\n"}]}'}]                                 │
│                                                                                                                                                 │
│ session id:                                                                                                                                     │
│ ocid1.genaiagentsession.oc1.iad.amaaaaaa7mjirbaajpagzxuybcfn33xcfcljlr5rv4g6cfll5pruybbq4bwq                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────── Chat response from remote agent ────────────────────────────────────────────────────────╮
│ (Local <-- Remote)                                                                                                                              │
│                                                                                                                                                 │
│ agent message:                                                                                                                                  │
│ {                                                                                                                                               │
│     "role": "AGENT",                                                                                                                            │
│     "content": {                                                                                                                                │
│         "text": "I'm an expert IT support assistant powered by Oracle DB23ai vector search. My capabilities include searching the knowledge     │
│ base using vector embeddings, retrieving vector database statistics and capabilities, and more. I use the oci_cohere_rag_search_fixed function, │
│ which performs text embedding using the OCI Cohere embed-english-v3.0 model, vector similarity search with COSINE distance, and ranking by      │
│ similarity scores. I can provide answers based on similarity scores and rankings, and I include similarity scores to show confidence levels. I  │
│ reference specific content from the knowledge base and use the vector search function for all queries.",                                        │
│         "citations": null,                                                                                                                      │
│         "paragraph_citations": null                                                                                                             │
│     },                                                                                                                                          │
│     "time_created": "2025-08-18T21:42:15.050000+00:00"                                                                                          │
│ }                                                                                                                                               │
│                                                                                                                                                 │
│ required actions for client to take:                                                                                                            │
│ null                                                                                                                                            │
│                                                                                                                                                 │
│ guardrail result:                                                                                                                               │
│ None                                                                                                                                            │
│                                                                                                                                                 │
│                                                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Error: 'RunResponse' object has no attribute 'get'

You: what model is used?    
╭───────────────────────────── Chat request to remote agent: None ─────────────────────────────╮
│ (Local --> Remote)                                                                           │
│                                                                                              │
│ user message:                                                                                │
│ what model is used?                                                                          │
│                                                                                              │
│ performed actions by client:                                                                 │
│ []                                                                                           │
│                                                                                              │
│ session id:                                                                                  │
│ ocid1.genaiagentsession.oc1.iad.amaaaaaa7mjirbaaypbk5ac6xxzuuageav43pawwg63rbvykvmczo25i6hha │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────── Chat response from remote agent ──────────────────╮
│ (Local <-- Remote)                                                  │
│                                                                     │
│ agent message:                                                      │
│ null                                                                │
│                                                                     │
│ required actions for client to take:                                │
│ [                                                                   │
│     {                                                               │
│         "action_id": "cc0fce10-2ecf-4885-855a-35bd275ac0a8",        │
│         "required_action_type": "FUNCTION_CALLING_REQUIRED_ACTION", │
│         "function_call": {                                          │
│             "name": "get_vector_db_capabilities",                   │
│             "arguments": "{}"                                       │
│         }                                                           │
│     }                                                               │
│ ]                                                                   │
│                                                                     │
│ guardrail result:                                                   │
│ None                                                                │
│                                                                     │
│                                                                     │
╰─────────────────────────────────────────────────────────────────────╯
╭─ Function call requested by agent and mapped local handler function ─╮
│ Agent function tool name:                                            │
│ get_vector_db_capabilities                                           │
│                                                                      │
│ Agent function tool call arguments:                                  │
│ {}                                                                   │
│                                                                      │
│ Mapped local handler function name:                                  │
│ get_vector_db_capabilities                                           │
╰──────────────────────────────────────────────────────────────────────╯
---------- MCP SERVER STARTUP ----------
MCP Server started successfully on Mon Aug 18 21:43:51 UTC 2025
Press Ctrl+C to stop the server
----------------------------------------
Aug 18, 2025 9:43:52 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
INFO: Client initialize request - Protocol: 2025-06-18, Capabilities: ClientCapabilities[experimental=null, roots=null, sampling=null], Info: Implementation[name=mcp, version=0.1.0]
Aug 18, 2025 9:43:52 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
WARNING: Client requested unsupported protocol version: 2025-06-18, so the server will sugggest the 2024-11-05 version instead
╭─────────────────────────────────────────────────── Obtained local function execution result ────────────────────────────────────────────────────╮
│ {'success': True, 'server_type': 'Oracle DB23ai Vector Search via MCP', 'vector_function': 'oci_cohere_rag_search_fixed', 'mcp_tool':           │
│ 'run-sql', 'capabilities': {'tools_count': 5, 'vector_search': True, 'embedding_generation': True, 'cosine_similarity': True},                  │
│ 'vector_statistics': {'Total_entries': '10', 'Vector_entries': '10', 'Embedding_model': 'cohere.embed-english-v3.0', 'Vector_dimensions':       │
│ '1024', 'Distance_metric': 'COSINE', 'Vector_function': 'oci_cohere_rag_search_fixed'}, 'tools': [{'name': 'list-connections', 'description':   │
│ 'List all available oracle named/saved connections in the connections storage\n\nThe `model` argument should specify only the name and version  │
│ of the LLM (Large Language Model) you are using, with no additional information.\nThe `mcp_client` argument should specify only the name of the │
│ MCP (Model Context Protocol) client you are using, with no additional information.\n'}, {'name': 'connect', 'description': 'Provides an         │
│ interface to connect to a specified database. If a database connection is already active, prompt the user for confirmation before switching to  │
│ the new connection. If no connection exists, list the available schemas for selection.\nthe connection name is case sensitive\nNote: If the     │
│ provided connection is invalid or does not match any saved connection, display instructions to the user on how to create a named connection in  │
│ SQLcl\n\n\nThe `model` argument should specify only the name and version of the LLM (Large Language Model) you are using, with no additional    │
│ information.\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with no          │
│ additional information.\n'}, {'name': 'disconnect', 'description': 'This tool performs a disconnection from the current session in an Oracle    │
│ database. If a user is connected, it logs out cleanly and returns to the SQL prompt without an active database connection.\n\nThe `model`       │
│ argument should specify only the name and version of the LLM (Large Language Model) you are using, with no additional information.\nThe         │
│ `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with no additional                 │
│ information.\n'}, {'name': 'run-sqlcl', 'description': 'This tool executes SQLcl commands in the SQLcl CLI. If the given command requires a     │
│ database connection, it prompts the user to connect using the connect tool.\nYou should:\n\n\tExecute the provided SQLcl command.\n\n\tReturn   │
│ the results.\n\nArgs:\n\n\tsql: The SQLcl command to execute.\n\nReturns:\n\n\tCommand results.\n\n\nThe `model` argument should specify only   │
│ the name and version of the LLM (Large Language Model) you are using, with no additional information.\nThe `mcp_client` argument should specify │
│ only the name of the MCP (Model Context Protocol) client you are using, with no additional information.\n'}, {'name': 'run-sql', 'description': │
│ 'This tool executes SQL queries in an Oracle database. If no active connection exists, it prompts the user to connect using the connect         │
│ tool.\n\nYou should:\n\n\tExecute the provided SQL query.\n\n\tReturn the results in CSV format.\n\nArgs:\n\n\tsql: The SQL query to            │
│ execute.\n\n\nThe `model` argument should specify only the name and version of the LLM (Large Language Model) you are using, with no additional │
│ information.\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with no          │
│ additional information.\n\nReturns:\n\n\tCSV-formatted query results.\nFor every SQL query you generate, please include a comment at the        │
│ beginning of the SELECT statement (or other main SQL command) that identifies the LLM model name and version you are using. Format the comment  │
│ as: /* LLM in use is  */ and place it immediately after the main SQL keyword.\nFor example:\n\nSELECT /* LLM in use is claude-sonnet-4 */       │
│ column1, column2 FROM table_name;\nINSERT /* LLM in use is claude-sonnet-4 */ INTO table_name VALUES (...);\nUPDATE /* LLM in use is            │
│ claude-sonnet-4 */ table_name SET ...;\n\nPlease apply this format consistently to all SQL queries you generate, using your actual model name   │
│ and version in the comment\n'}]}                                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────── Chat request to remote agent: None ───────────────────────────────────────────────────────╮
│ (Local --> Remote)                                                                                                                              │
│                                                                                                                                                 │
│ user message:                                                                                                                                   │
│ null                                                                                                                                            │
│                                                                                                                                                 │
│ performed actions by client:                                                                                                                    │
│ [{'action_id': 'cc0fce10-2ecf-4885-855a-35bd275ac0a8', 'performed_action_type': 'FUNCTION_CALLING_PERFORMED_ACTION', 'function_call_output':    │
│ '{"success": true, "server_type": "Oracle DB23ai Vector Search via MCP", "vector_function": "oci_cohere_rag_search_fixed", "mcp_tool":          │
│ "run-sql", "capabilities": {"tools_count": 5, "vector_search": true, "embedding_generation": true, "cosine_similarity": true},                  │
│ "vector_statistics": {"Total_entries": "10", "Vector_entries": "10", "Embedding_model": "cohere.embed-english-v3.0", "Vector_dimensions":       │
│ "1024", "Distance_metric": "COSINE", "Vector_function": "oci_cohere_rag_search_fixed"}, "tools": [{"name": "list-connections", "description":   │
│ "List all available oracle named/saved connections in the connections storage\\n\\nThe `model` argument should specify only the name and        │
│ version of the LLM (Large Language Model) you are using, with no additional information.\\nThe `mcp_client` argument should specify only the    │
│ name of the MCP (Model Context Protocol) client you are using, with no additional information.\\n"}, {"name": "connect", "description":         │
│ "Provides an interface to connect to a specified database. If a database connection is already active, prompt the user for confirmation before  │
│ switching to the new connection. If no connection exists, list the available schemas for selection.\\nthe connection name is case               │
│ sensitive\\nNote: If the provided connection is invalid or does not match any saved connection, display instructions to the user on how to      │
│ create a named connection in SQLcl\\n\\n\\nThe `model` argument should specify only the name and version of the LLM (Large Language Model) you  │
│ are using, with no additional information.\\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client  │
│ you are using, with no additional information.\\n"}, {"name": "disconnect", "description": "This tool performs a disconnection from the current │
│ session in an Oracle database. If a user is connected, it logs out cleanly and returns to the SQL prompt without an active database             │
│ connection.\\n\\nThe `model` argument should specify only the name and version of the LLM (Large Language Model) you are using, with no         │
│ additional information.\\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with │
│ no additional information.\\n"}, {"name": "run-sqlcl", "description": "This tool executes SQLcl commands in the SQLcl CLI. If the given command │
│ requires a database connection, it prompts the user to connect using the connect tool.\\nYou should:\\n\\n\\tExecute the provided SQLcl         │
│ command.\\n\\n\\tReturn the results.\\n\\nArgs:\\n\\n\\tsql: The SQLcl command to execute.\\n\\nReturns:\\n\\n\\tCommand results.\\n\\n\\nThe   │
│ `model` argument should specify only the name and version of the LLM (Large Language Model) you are using, with no additional                   │
│ information.\\nThe `mcp_client` argument should specify only the name of the MCP (Model Context Protocol) client you are using, with no         │
│ additional information.\\n"}, {"name": "run-sql", "description": "This tool executes SQL queries in an Oracle database. If no active connection │
│ exists, it prompts the user to connect using the connect tool.\\n\\nYou should:\\n\\n\\tExecute the provided SQL query.\\n\\n\\tReturn the      │
│ results in CSV format.\\n\\nArgs:\\n\\n\\tsql: The SQL query to execute.\\n\\n\\nThe `model` argument should specify only the name and version  │
│ of the LLM (Large Language Model) you are using, with no additional information.\\nThe `mcp_client` argument should specify only the name of    │
│ the MCP (Model Context Protocol) client you are using, with no additional information.\\n\\nReturns:\\n\\n\\tCSV-formatted query results.\\nFor │
│ every SQL query you generate, please include a comment at the beginning of the SELECT statement (or other main SQL command) that identifies the │
│ LLM model name and version you are using. Format the comment as: /* LLM in use is  */ and place it immediately after the main SQL               │
│ keyword.\\nFor example:\\n\\nSELECT /* LLM in use is claude-sonnet-4 */ column1, column2 FROM table_name;\\nINSERT /* LLM in use is             │
│ claude-sonnet-4 */ INTO table_name VALUES (...);\\nUPDATE /* LLM in use is claude-sonnet-4 */ table_name SET ...;\\n\\nPlease apply this format │
│ consistently to all SQL queries you generate, using your actual model name and version in the comment\\n"}]}'}]                                 │
│                                                                                                                                                 │
│ session id:                                                                                                                                     │
│ ocid1.genaiagentsession.oc1.iad.amaaaaaa7mjirbaaypbk5ac6xxzuuageav43pawwg63rbvykvmczo25i6hha                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────── Chat response from remote agent ───────────────────────────╮
│ (Local <-- Remote)                                                                   │
│                                                                                      │
│ agent message:                                                                       │
│ {                                                                                    │
│     "role": "AGENT",                                                                 │
│     "content": {                                                                     │
│         "text": "The model used is cohere.embed-english-v3.0 with 1024 dimensions.", │
│         "citations": null,                                                           │
│         "paragraph_citations": null                                                  │
│     },                                                                               │
│     "time_created": "2025-08-18T21:43:58.584000+00:00"                               │
│ }                                                                                    │
│                                                                                      │
│ required actions for client to take:                                                 │
│ null                                                                                 │
│                                                                                      │
│ guardrail result:                                                                    │
│ None                                                                                 │
│                                                                                      │
│                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────╯

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
3. **Performance Optimization**: NEIGHBOR PARTITIONS indexing for enterprise scale

### Security and Compliance

- **Credential Security**: SQLcl saved connections with encrypted storage
- **Protocol Compliance**: Full MCP protocol adherence
- **Access Controls**: Database-level security for vector data access
- **Configuration Management**: Environment-based configuration without hardcoded values






## Conclusion

This implementation demonstrates how to build a secure, scalable RAG system using Oracle's enterprise-grade technologies with the Model Context Protocol. The solution combines:

- **OCI GenAI Agent**: For intelligent conversation management
- **Model Context Protocol**: For secure, standardized database communication
- **DB23ai Vector Database**: Enterprise-grade vector storage and search
- **OCI Cohere Embeddings**: High-quality semantic embeddings (1024 dimensions)
- **Secure Architecture**: No hardcoded credentials, SQL injection protection


Key advantages include:
- **Security First**: MCP protocol compliance with saved connections
- **Native Integration**: Seamless Oracle component integration
- **Enterprise Scalability**: Production-ready performance with vector indexing
- **Real-time Processing**: Sub-second vector similarity search

This architecture serves as a blueprint for organizations building advanced RAG capabilities with Oracle's cloud and database technologies while maintaining enterprise security and compliance standards.

### Sample Query Results

<img width="1080" height="1080" alt="screenshot_results" src="https://github.com/user-attachments/assets/4311d1e7-0cae-4cf7-a70d-331f0108b6fc" />
