# Oracle DB23ai MCP RAG Agent: Complete Implementation Guide

## Overview

This guide demonstrates how to build a complete RAG (Retrieval-Augmented Generation) agent using Oracle DB23ai's native Model Context Protocol (MCP) capabilities. The implementation provides real-time vector similarity search through OCI GenAI Agents with direct MCP integration.

## Architecture

```
OCI GenAI Agent → Python Tools → MCP Client → Oracle DB23ai MCP Server → Vector Database
```

- **Frontend**: OCI GenAI Agent with natural language interface
- **Backend**: Oracle DB23ai with native MCP support (`sql -mcp`)
- **Vector Search**: `oci_cohere_rag_search_fixed` function with OCI Cohere embeddings
- **Protocol**: Model Context Protocol for standardized AI-database communication

## Step-by-Step Implementation

### Step 1: Set Up Environment

```bash
# Ensure Oracle wallet is configured
export TNS_ADMIN=/home/ubuntu/wallet-db

# Verify Java 17+ is available (required for MCP server)
java -version

# Install dependencies
uv add mcp oci-addons-adk
```

### Step 2: Create MCP Client Class

```python
class DB23aiMCPClient:
    """MCP client for connecting to DB23ai MCP server with saved connections"""
    
    def __init__(self):
        self.tns_admin = '/home/ubuntu/wallet-db'
        self.connection_name = "mcp_saved"  # Use saved connection
```

**Key Insight**: Use saved connections instead of direct credentials for proper database connectivity.

### Step 3: Implement MCP Connection Method

```python
async def connect_and_search(self, query: str, top_k: int = 3, similarity_threshold: float = 0.6):
    """Connect to MCP server and perform search using native MCP tools"""
    env = os.environ.copy()
    env['TNS_ADMIN'] = self.tns_admin
    
    # Start MCP server without credentials, then use connect tool
    server_params = StdioServerParameters(
        command="./sqlcl/bin/sql",
        args=["-mcp"],  # No credentials here!
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
```

**Critical Fix**: Start MCP server with `args=["-mcp"]` only, then use the connect tool with saved connection name.

### Step 4: Execute Vector Search via MCP

```python
# Use Oracle's native MCP run-sql tool with vector function
safe_query = query.replace("'", "''")

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
    l_cursor := oci_cohere_rag_search_fixed('{safe_query}', {top_k}, {similarity_threshold});
    
    LOOP
        FETCH l_cursor INTO l_topic, l_text, l_question, l_answer, l_similarity, l_rank;
        EXIT WHEN l_cursor%NOTFOUND;
        
        l_count := l_count + 1;
        DBMS_OUTPUT.PUT_LINE('RESULT_' || l_count || ':');
        DBMS_OUTPUT.PUT_LINE('Topic: ' || l_topic);
        DBMS_OUTPUT.PUT_LINE('Content: ' || SUBSTR(l_text, 1, 200));
        DBMS_OUTPUT.PUT_LINE('Answer: ' || SUBSTR(l_answer, 1, 300));
        DBMS_OUTPUT.PUT_LINE('Similarity: ' || l_similarity);
    END LOOP;
    
    DBMS_OUTPUT.PUT_LINE('Total results: ' || l_count);
    CLOSE l_cursor;
END;
/"""

# Execute via Oracle's native MCP run-sql tool
result = await session.call_tool(
    "run-sql",
    arguments={"sql": vector_sql}
)
```

### Step 5: Parse MCP Results

```python
def _parse_mcp_results(self, mcp_output: str) -> List[Dict[str, Any]]:
    """Parse vector RAG results from MCP run-sql tool DBMS_OUTPUT"""
    results = []
    current_result = {}
    
    # Extract the actual text content from MCP format
    if "text='" in mcp_output:
        start = mcp_output.find("text='") + 6
        end = mcp_output.rfind("' annotations=None")
        if end == -1:
            end = len(mcp_output)
        actual_text = mcp_output[start:end]
    else:
        actual_text = mcp_output
    
    # Parse DBMS_OUTPUT format
    lines = actual_text.split('\\\\n') if '\\\\n' in actual_text else actual_text.split('\\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('RESULT_') and line.endswith(':'):
            current_result = {}
        elif ':' in line and current_result is not None:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'Topic':
                current_result['topic'] = value
            elif key == 'Content':
                current_result['content'] = value
            elif key == 'Answer':
                current_result['solution'] = value
            elif key == 'Similarity':
                current_result['similarity_score'] = float(value)
    
    return results
```

### Step 6: Create Tool Functions

```python
@tool
def search_knowledge_base_vector(query: str, top_k: int = 3, similarity_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Search DB23ai knowledge base using direct MCP connection with native tools.
    
    Returns:
        Dict[str, Any]: Direct MCP vector search results with cosine similarity scores
    """
    return asyncio.run(mcp_client.connect_and_search(query, top_k, similarity_threshold))

@tool  
def get_db23ai_capabilities() -> Dict[str, Any]:
    """Get DB23ai MCP server capabilities and configuration via direct connection."""
    return asyncio.run(mcp_client.get_mcp_server_capabilities())
```

### Step 7: Configure OCI GenAI Agent

```python
class FinalDB23aiMCPRAGAgent:
    """Final OCI GenAI Agent with DB23ai MCP integration"""
    
    def __init__(self, agent_endpoint_id: str = None):
        self.agent_endpoint_id = agent_endpoint_id or "ocid1.genaiagentendpoint.oc1.iad.amaaaaaaieosruaaxpzdp7gwdk3hd5selesyszveooa3huxi3r2y2m3tsjkq"
        
        self.instructions = """
        You are an expert IT support assistant powered by Oracle DB23ai's native MCP protocol.
        
        Your capabilities use advanced vector search technology via direct MCP connection:
        - search_knowledge_base_vector: Direct MCP vector search with native tools
        - get_db23ai_capabilities: Direct MCP database configuration and statistics
        
        Technical specifications:
        - Protocol: Model Context Protocol with saved connections
        - Connection: sql -mcp + connect tool
        - Vector Function: oci_cohere_rag_search_fixed
        - Vector Model: OCI Cohere embed-english-v3.0 (1024 dimensions)
        - Search Method: Cosine similarity with NEIGHBOR_PARTITIONS indexing
        """
        
        self.tools = [search_knowledge_base_vector, get_db23ai_capabilities]
```

### Step 8: Test the Implementation

```python
# Test MCP capabilities
info = get_db23ai_capabilities()
print(f"MCP Server: {info.get('server_type')}")
print(f"Tools Available: {info['capabilities']['tools_count']}")

# Test vector search
result = search_knowledge_base_vector("setup conference call", 2)
if result.get('success') and result.get('results_count', 0) > 0:
    for res in result['results']:
        print(f"Topic: {res.get('topic')}")
        print(f"Score: {res.get('similarity_score')}")
        print(f"Solution: {res.get('solution')[:100]}...")
```

## Key Technical Insights

### 1. MCP Connection Pattern

❌ **Wrong**: Start with credentials in args
```python
args=["-mcp", "ADMIN/\"password\"@dsn"]  # Returns "Connection not established"
```

✅ **Correct**: Start without credentials, use connect tool
```python
args=["-mcp"]  # Start server
await session.call_tool("connect", {"connection_name": "mcp_saved"})  # Connect
```

### 2. Oracle's Native MCP Tools

The Oracle DB23ai MCP server provides these 5 native tools:
1. **list-connections**: List saved connections
2. **connect**: Connect to database
3. **disconnect**: Disconnect from database  
4. **run-sqlcl**: Execute SQLcl commands
5. **run-sql**: Execute SQL queries (used for vector search)

### 3. Vector Search Implementation

- Uses `oci_cohere_rag_search_fixed()` PL/SQL function
- Returns structured results via DBMS_OUTPUT
- Cosine similarity scoring with configurable threshold
- NEIGHBOR_PARTITIONS indexing for performance

## Usage in OCI GenAI Agent

### Two-Step Function Execution

1. **User Input**: "setup conference call"
2. **Function Call Display**: Shows `search_knowledge_base_vector` parameters
3. **User Approval**: Press Enter to approve function execution
4. **Real Results**: Returns actual database content like "Setting Up a Conference Call on Cisco Webex"

### Sample Results

```json
{
  "success": true,
  "results_count": 1,
  "results": [
    {
      "topic": "Setting Up a Conference Call on Cisco Webex",
      "similarity_score": 0.565914,
      "solution": "To set up a conference call on Cisco Webex, follow these steps...",
      "search_method": "MCP_Vector_Search"
    }
  ],
  "technology": "Oracle DB23ai MCP Protocol"
}
```

## Troubleshooting

### Common Issues

1. **"Connection not established"**: Use saved connection approach, not direct credentials
2. **Empty results**: Check MCP output parsing and ensure database connection is active
3. **Java version errors**: Ensure Java 17+ is installed for MCP server
4. **TNS_ADMIN errors**: Verify Oracle wallet path is correct

### Debug Commands

```python
# Test MCP connection
result = search_knowledge_base_vector("test query", 1)
print(f"Success: {result.get('success')}")
print(f"Results: {result.get('results_count', 0)}")

# Check MCP capabilities  
info = get_db23ai_capabilities()
print(f"Available tools: {len(info.get('tools', []))}")
```

## Conclusion

This implementation provides a complete, production-ready RAG agent using Oracle DB23ai's native MCP protocol. The key breakthrough was understanding the correct MCP connection pattern using saved connections rather than direct credentials.

The system automatically manages MCP server lifecycle, provides real-time vector similarity search, and integrates seamlessly with OCI GenAI Agents for natural language interactions with enterprise knowledge bases.

## Benefits

- ✅ **Native MCP Protocol**: Direct integration with Oracle's MCP server
- ✅ **Automatic Server Management**: No manual MCP server startup required
- ✅ **Real Database Results**: Actual vector search results from Oracle DB23ai
- ✅ **Production Ready**: Error handling, connection management, result parsing
- ✅ **Standardized Interface**: Model Context Protocol for AI-database communication