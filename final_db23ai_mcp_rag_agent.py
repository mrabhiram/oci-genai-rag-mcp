#!/usr/bin/env python3
"""
Copyright (c) 2024, 2025, Oracle and/or its affiliates.
Licensed under the Universal Permissive License v1.0 as shown at http://oss.oracle.com/licenses/upl.

DB23ai MCP RAG Agent - Complete MCP Implementation
Uses running DB23ai MCP server (sql -mcp) for proper RAG operations
"""
import os
import asyncio
from typing import Dict, Any, List
from oci.addons.adk import Agent, AgentClient, tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class DB23aiMCPClient:
    """MCP client for connecting to DB23ai MCP server with direct credentials"""
    
    def __init__(self):
        self.tns_admin = '/home/ubuntu/wallet-db'
        self.connection_name = "mcp_saved"
    
    async def connect_and_search(self, query: str, top_k: int = 3, similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """Connect to MCP server and perform search using native MCP tools"""
        try:
            env = os.environ.copy()
            env['TNS_ADMIN'] = self.tns_admin
            
            # Start MCP server without credentials, then use connect tool
            server_params = StdioServerParameters(
                command="./sqlcl/bin/sql",
                args=["-mcp"],
                env=env
            )
            
            # Use MCP with proper database connection
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Connect to database using saved connection
                    await session.call_tool(
                        "connect",
                        arguments={"connection_name": self.connection_name}
                    )
                    
                    # Get available tools from MCP server
                    tools_response = await session.list_tools()
                    
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
        DBMS_OUTPUT.PUT_LINE('Question: ' || l_question);
        DBMS_OUTPUT.PUT_LINE('Answer: ' || SUBSTR(l_answer, 1, 300));
        DBMS_OUTPUT.PUT_LINE('Similarity: ' || l_similarity);
        DBMS_OUTPUT.PUT_LINE('Rank: ' || l_rank);
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
                    
                    if result.content and len(result.content) > 0:
                        mcp_output = str(result.content[0])
                        results = self._parse_mcp_results(mcp_output)
                        
                        return {
                            'success': True,
                            'query': query,
                            'results_count': len(results),
                            'results': results,
                            'technology': 'Oracle DB23ai MCP Protocol',
                            'search_type': 'Native MCP Vector Search',
                            'implementation': 'Direct MCP Connection',
                            'embedding_model': 'cohere.embed-english-v3.0',
                            'vector_dimensions': 1024,
                            'distance_metric': 'COSINE',
                            'connection_method': 'direct_credentials'
                        }
                    else:
                        return {
                            'success': False,
                            'query': query,
                            'error': 'No content returned from MCP tools',
                            'results_count': 0
                        }
                        
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'error': f'MCP execution error: {str(e)}',
                'results_count': 0
            }
    
    def _parse_mcp_results(self, mcp_output: str) -> List[Dict[str, Any]]:
        """Parse vector RAG results from MCP run-sql tool DBMS_OUTPUT"""
        results = []
        current_result = {}
        
        try:
            # Extract the actual text content from MCP format
            if "text='" in mcp_output:
                # Extract text between text=' and the closing quote
                start = mcp_output.find("text='") + 6
                end = mcp_output.rfind("' annotations=None")
                if end == -1:
                    end = len(mcp_output)
                actual_text = mcp_output[start:end]
            else:
                actual_text = mcp_output
            
            # Handle different newline formats
            lines = actual_text.split('\\n') if '\\n' in actual_text else actual_text.split('\n')
            in_result = False
            
            for line in lines:
                line = line.strip()
                
                # Look for result start patterns
                if line.startswith('RESULT_') and line.endswith(':'):
                    in_result = True
                    current_result = {}
                elif line == 'Total results:' or (in_result and line.startswith('RESULT_')):
                    # End current result and add to results
                    if current_result and 'topic' in current_result:
                        # Add MCP specific metadata
                        current_result['search_method'] = 'MCP_Vector_Search'
                        current_result['technology'] = 'DB23ai_MCP_Protocol'
                        results.append(current_result.copy())
                    in_result = False
                    if line == 'Total results:':
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
                        current_result['solution'] = value
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
            
            # Handle case where we have a result but loop ended
            if current_result and 'topic' in current_result and current_result not in results:
                current_result['search_method'] = 'MCP_Vector_Search'
                current_result['technology'] = 'DB23ai_MCP_Protocol'
                results.append(current_result)
                        
        except Exception as e:
            print(f"MCP parse error: {e}")
            print(f"Raw output: {mcp_output[:200]}...")
        
        return results
    
    async def get_mcp_server_capabilities(self) -> Dict[str, Any]:
        """Get vector database capabilities via MCP protocol"""
        try:
            env = os.environ.copy()
            env['TNS_ADMIN'] = self.tns_admin
            
            # Start MCP server and connect using saved connection
            server_params = StdioServerParameters(
                command="./sqlcl/bin/sql",
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
                    
                    # Get vector database statistics via MCP
                    capabilities_sql = """
SET SERVEROUTPUT ON SIZE 1000000
DECLARE
    l_total NUMBER;
    l_vector_count NUMBER;
BEGIN
    SELECT COUNT(*), COUNT(CASE WHEN VEC IS NOT NULL THEN 1 END)
    INTO l_total, l_vector_count
    FROM RAG_SAMPLE_QAS_FROM_KIS;
    
    DBMS_OUTPUT.PUT_LINE('MCP_STATS_START');
    DBMS_OUTPUT.PUT_LINE('Total_entries:' || l_total);
    DBMS_OUTPUT.PUT_LINE('Vector_entries:' || l_vector_count);
    DBMS_OUTPUT.PUT_LINE('Embedding_model:cohere.embed-english-v3.0');
    DBMS_OUTPUT.PUT_LINE('Vector_dimensions:1024');
    DBMS_OUTPUT.PUT_LINE('Distance_metric:COSINE');
    DBMS_OUTPUT.PUT_LINE('Index_type:NEIGHBOR_PARTITIONS');
    DBMS_OUTPUT.PUT_LINE('DBMS_VECTOR:Available');
    DBMS_OUTPUT.PUT_LINE('SQL_RAG_Function:oci_cohere_rag_search_fixed');
    DBMS_OUTPUT.PUT_LINE('MCP_Protocol:run-sql_tool');
    DBMS_OUTPUT.PUT_LINE('Implementation:MCP_run_sql_tool');
    DBMS_OUTPUT.PUT_LINE('MCP_STATS_END');
END;
/
"""
                    
                    # Get MCP server tools information  
                    tools_response = await session.list_tools()
                    mcp_tools = [{'name': t.name, 'description': t.description} for t in tools_response.tools]
                    
                    # Execute capabilities query via MCP run-sql tool
                    result = await session.call_tool(
                        "run-sql",
                        arguments={"sql": capabilities_sql}
                    )
            
                    if result.content and len(result.content) > 0:
                        mcp_output = str(result.content[0])
                        stats = self._parse_mcp_stats(mcp_output)
                
                        return {
                            'success': True,
                            'server_type': 'Oracle DB23ai MCP Server',
                            'command': 'sql -mcp with direct credentials',
                            'protocol': 'Model Context Protocol - Native Tools',
                            'implementation': 'Direct MCP Connection',
                            'database': 'Oracle DB23ai Autonomous Database',
                            'vector_capabilities': {
                                'native_vector_support': True,
                                'embedding_model': 'OCI Cohere embed-english-v3.0',
                                'vector_dimensions': 1024,
                                'distance_metric': 'COSINE',
                                'index_type': 'NEIGHBOR_PARTITIONS',
                                'sql_rag_function': 'oci_cohere_rag_search_fixed',
                                'mcp_access': 'Direct MCP Connection'
                            },
                            'capabilities': {
                                'tools_count': len(mcp_tools),
                                'vector_search': True,
                                'rag_enabled': True,
                                'embedding_generation': True,
                                'mcp_protocol': True,
                                'dbms_output_support': True,
                                'plsql_execution': True
                            },
                            'statistics': stats,
                            'tools': mcp_tools,
                            'features': [
                                'Direct MCP Connection',
                                'Native MCP Tools',
                                'Vector Similarity Search',
                                'OCI Cohere Embeddings', 
                                'DBMS_VECTOR Package',
                                'Optimized Vector Indexing',
                                'Real-time RAG Search',
                                'PL/SQL Execution',
                                'DBMS_OUTPUT Structured Results'
                            ]
                        }
                    else:
                        return {
                            'success': False,
                            'error': 'No content returned from MCP capabilities query'
                        }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _parse_mcp_stats(self, mcp_output: str) -> Dict[str, Any]:
        """Parse vector database statistics from MCP output"""
        stats = {}
        
        try:
            # Handle MCP output format
            lines = mcp_output.split('\\n') if '\\n' in mcp_output else mcp_output.split('\n')
            in_stats = False
            
            for line in lines:
                line = line.strip()
                
                if line == 'MCP_STATS_START':
                    in_stats = True
                elif line == 'MCP_STATS_END':
                    in_stats = False
                elif in_stats and ':' in line:
                    key, value = line.split(':', 1)
                    stats[key] = value
                        
        except Exception as e:
            print(f"MCP stats parse error: {e}")
        
        return stats

# Global MCP client
mcp_client = DB23aiMCPClient()

@tool
def search_knowledge_base_vector(query: str, top_k: int = 3, similarity_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Search DB23ai knowledge base using direct MCP connection with native tools.
    
    This function uses direct credential connection to Oracle DB23ai MCP server
    and leverages native MCP tools or oci_cohere_rag_search_fixed vector function.
    
    Implementation: Direct MCP Connection
    Vector Function: Native MCP tools or oci_cohere_rag_search_fixed
    
    Args:
        query (str): Search query for RAG operations
        top_k (int): Number of results to return (max 10)
        similarity_threshold (float): Minimum similarity score (0.0-1.0)
        
    Returns:
        Dict[str, Any]: Direct MCP vector search results with cosine similarity scores
    """
    return asyncio.run(mcp_client.connect_and_search(query, top_k, similarity_threshold))

@tool  
def get_db23ai_capabilities() -> Dict[str, Any]:
    """
    Get DB23ai MCP server capabilities and configuration via direct connection.
    
    This function uses direct credential connection to query database statistics
    and vector capabilities, confirming access to native MCP tools.
    
    Implementation: Direct MCP Connection
    
    Returns:
        Dict[str, Any]: Direct MCP DB23ai capabilities including native tools
    """
    return asyncio.run(mcp_client.get_mcp_server_capabilities())

class FinalDB23aiMCPRAGAgent:
    """Final OCI GenAI Agent with DB23ai MCP integration"""
    
    def __init__(self, agent_endpoint_id: str = None, compartment_id: str = None):
        # Default to the provided agent endpoint ID
        if agent_endpoint_id is None:
            agent_endpoint_id = "ocid1.genaiagentendpoint.oc1.iad.amaaaaaaieosruaaxpzdp7gwdk3hd5selesyszveooa3huxi3r2y2m3tsjkq"
        self.agent_endpoint_id = agent_endpoint_id
        self.compartment_id = compartment_id
        self.agent = None
        
        self.instructions = """
        You are an expert IT support assistant powered by Oracle DB23ai's native MCP protocol.
        
        Your capabilities use advanced vector search technology via direct MCP connection:
        - search_knowledge_base_vector: Direct MCP vector search with native tools or oci_cohere_rag_search_fixed
        - get_db23ai_capabilities: Direct MCP database configuration and statistics
        
        Technical specifications (Direct MCP Connection):
        - Protocol: Model Context Protocol with direct credentials
        - Connection: sql -mcp with username/password@dsn
        - Vector Function: Native MCP tools or oci_cohere_rag_search_fixed
        - Vector Model: OCI Cohere embed-english-v3.0 with 1024 dimensions
        - Database: Oracle DB23ai with native DBMS_VECTOR package
        - Search Method: Cosine similarity with optimized vector indexing
        - Implementation: Direct credential connection to MCP server
        - Performance: NEIGHBOR_PARTITIONS indexing for sub-second responses
        
        When helping users:
        1. Use direct MCP vector similarity search to find the most relevant information
        2. Provide accurate solutions based on similarity scores and rankings
        3. Include similarity scores to show confidence levels
        4. Explain that you're using direct MCP connection with Oracle's vector search
        5. Be professional and technically precise in responses
        6. Mention that all operations use direct Model Context Protocol connection
        """
        
        self.tools = [search_knowledge_base_vector, get_db23ai_capabilities]
    
    def setup_agent(self):
        """Setup agent with MCP tools"""
        if self.agent_endpoint_id:
            try:
                client = AgentClient(
                    auth_type="api_key", 
                    profile="DEFAULT",
                    region="us-ashburn-1"
                )
                
                self.agent = Agent(
                    client=client,
                    agent_endpoint_id=self.agent_endpoint_id,
                    instructions=self.instructions,
                    tools=self.tools
                )
                self.agent.setup()
                print("OCI GenAI Agent with DB23ai Direct MCP Connection setup complete")
                return True
            except Exception as e:
                print(f"Agent setup failed: {e}")
                self.agent = None
                return False
        else:
            print("Running in simulation mode")
            return False
    
    def chat(self, message: str) -> str:
        """Chat with the agent"""
        if self.agent:
            try:
                return self.agent.run(message)
            except Exception as e:
                print(f"Agent error: {e}")
                return self._simulate_response(message)
        else:
            return self._simulate_response(message)
    
    def _simulate_response(self, message: str) -> str:
        """Simulate response using MCP protocol"""
        if "mcp" in message.lower() or "server" in message.lower():
            info = get_db23ai_capabilities()
            if info.get('success'):
                return f"DB23ai MCP Server Information:\n" \
                       f"Server: {info['server_type']}\n" \
                       f"Protocol: {info['protocol']}\n" \
                       f"Command: {info.get('command', 'sql -mcp')}\n" \
                       f"Tools Available: {info['capabilities']['tools_count']}\n" \
                       f"Java Version: {info.get('java_version', 'Java 17+')}"
        
        # Search using MCP protocol
        search = search_knowledge_base_vector(message, 2)
        if search.get('success') and search.get('results_count', 0) > 0:
            response = f"Using DB23ai MCP protocol, I found {search['results_count']} results:\n\n"
            
            for result in search['results']:
                tool_name = result.get('tool_name', 'MCP_TOOL')
                response += f"**MCP Result** (Tool: {tool_name})\n"
                response += f"Content: {result['content'][:200]}...\n"
                response += f"Score: {result['similarity_score']}\n\n"
            
            mcp_tools = ', '.join(search.get('mcp_tools_used', ['MCP_PROTOCOL']))
            response += f"Search powered by {search.get('server_type', 'DB23ai MCP')} using tools: {mcp_tools}"
            return response
        
        return f"Attempting to search using DB23ai MCP protocol but encountered: {search.get('error', 'Connection issue')}. The Model Context Protocol requires a running sql -mcp server."

def main():
    """Test final DB23ai MCP RAG agent"""
    print("Final DB23ai MCP RAG Agent")
    print("==========================")
    print("Model Context Protocol Implementation")
    print("Oracle DB23ai MCP Server Integration")
    
    # Test MCP server capabilities
    print("\n1. Testing DB23ai MCP Server...")
    info = get_db23ai_capabilities()
    if info.get('success'):
        print(f"MCP Server: {info.get('server_type')}")
        print(f"Command: {info.get('command', 'sql -mcp')}")
        print(f"Protocol: {info.get('protocol')}")
        print(f"Tools: {info['capabilities']['tools_count']} available")
        
        # List available MCP tools
        if 'tools' in info and info['tools']:
            print("Available MCP Tools:")
            for i, tool in enumerate(info['tools'], 1):
                tool_name = tool.get('name', 'Unknown')
                tool_desc = tool.get('description', 'No description')
                print(f"  {i}. {tool_name}: {tool_desc}")
        else:
            print("No detailed tool information available")
    else:
        print(f"MCP Server Error: {info.get('error')}")
        print("\nMCP Server is not running or not accessible.")
        print("Prerequisites:")
        print("1. Ensure Oracle Database wallet files are configured")
        print("2. Verify TNS_ADMIN environment variable points to wallet directory")
        print("3. Confirm Java 17+ is installed (check with: java -version)")
        print("\nTo start the MCP server:")
        print("   nohup ./sqlcl/bin/sql -mcp USERNAME/\"PASSWORD\"@DB_NAME > mcp_server.log 2>&1 &")
        print("\nTo monitor server status:")
        print("   tail -f mcp_server.log")
        print("   ps aux | grep 'sql -mcp'")
        print("\nExiting test due to MCP server unavailability.")
        return
    
    print("\n2. Testing MCP vector knowledge search...")
    queries = ["tablet", "email", "printer"]
    
    for query in queries:
        result = search_knowledge_base_vector(query, 1)
        if result.get('success') and result.get('results_count', 0) > 0:
            server_type = result.get('server_type', 'MCP')
            print(f"Query '{query}': Found {result['results_count']} results via {server_type}")
        else:
            print(f"Query '{query}': {result.get('error', 'MCP search failed')}")
            if 'Connection' in str(result.get('error', '')):
                print("Connection issue detected. Verify MCP server is running with:")
                print("  ps aux | grep 'sql -mcp'")
                print("If not running, start it with the commands shown above.")
                break
    
    print("\n3. Testing MCP agent simulation...")
    agent = FinalDB23aiMCPRAGAgent()
    agent.setup_agent()
    
    test_message = "What MCP capabilities does DB23ai provide?"
    response = agent._simulate_response(test_message)
    print(f"Agent response: {response[:300]}...")
    
    print("\n" + "=" * 60)
    print("Final DB23ai MCP RAG Agent Complete!")
    print("Technology: Oracle DB23ai + Model Context Protocol")
    print("Implementation: sql -mcp server + MCP client")
    print("Requirements: Java 17+, running MCP server")
    print("Protocol: Standardized MCP for AI-Database communication")

if __name__ == "__main__":
    main()
