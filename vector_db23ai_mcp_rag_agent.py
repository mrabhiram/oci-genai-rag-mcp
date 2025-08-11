#!/usr/bin/env python3
"""
Copyright (c) 2024, 2025, Oracle and/or its affiliates.
Licensed under the Universal Permissive License v1.0 as shown at http://oss.oracle.com/licenses/upl.

Vector DB23ai MCP RAG Agent - Uses vector search function via MCP run-sql tool
Implements oci_cohere_rag_search_fixed function through MCP protocol
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
                    
                    # Prepare vector search SQL with oci_cohere_rag_search_fixed function
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
                # Extract text content from MCP format
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
                    # Start new result
                    if current_result and 'topic' in current_result:
                        results.append(current_result.copy())
                    current_result = {}
                    in_result = True
                    
                elif line == '---' or line.startswith('Total results:'):
                    # End current result
                    if current_result and 'topic' in current_result:
                        results.append(current_result.copy())
                    in_result = False
                    if line.startswith('Total results:'):
                        break
                        
                elif in_result and ':' in line:
                    # Parse result fields
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
            print(f"Raw MCP output: {mcp_output[:500]}...")
        
        return results
    
    async def get_vector_capabilities(self) -> Dict[str, Any]:
        """Get vector database capabilities and statistics"""
        try:
            env = os.environ.copy()
            env['TNS_ADMIN'] = self.tns_admin
            
            server_params = StdioServerParameters(
                command=self.config.mcp.sqlcl_path,
                args=["-mcp"],
                env=env
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Connect to database
                    await session.call_tool(
                        "connect",
                        arguments={"connection_name": self.connection_name}
                    )
                    
                    # Get MCP capabilities
                    tools_response = await session.list_tools()
                    
                    # Get vector database statistics
                    stats_sql = """
SET SERVEROUTPUT ON SIZE 1000000
DECLARE
    l_total NUMBER;
    l_vector_count NUMBER;
BEGIN
    SELECT COUNT(*), COUNT(CASE WHEN VEC IS NOT NULL THEN 1 END)
    INTO l_total, l_vector_count
    FROM RAG_SAMPLE_QAS_FROM_KIS;
    
    DBMS_OUTPUT.PUT_LINE('VECTOR_STATS_START');
    DBMS_OUTPUT.PUT_LINE('Total_entries:' || l_total);
    DBMS_OUTPUT.PUT_LINE('Vector_entries:' || l_vector_count);
    DBMS_OUTPUT.PUT_LINE('Embedding_model:cohere.embed-english-v3.0');
    DBMS_OUTPUT.PUT_LINE('Vector_dimensions:1024');
    DBMS_OUTPUT.PUT_LINE('Distance_metric:COSINE');
    DBMS_OUTPUT.PUT_LINE('Vector_function:oci_cohere_rag_search_fixed');
    DBMS_OUTPUT.PUT_LINE('VECTOR_STATS_END');
END;
/"""
                    
                    result = await session.call_tool("run-sql", arguments={"sql": stats_sql})
                    stats = {}
                    
                    if result.content:
                        mcp_output = str(result.content[0])
                        stats = self._parse_stats(mcp_output)
                    
                    return {
                        'success': True,
                        'server_type': 'Oracle DB23ai Vector Search via MCP',
                        'vector_function': 'oci_cohere_rag_search_fixed',
                        'mcp_tool': 'run-sql',
                        'capabilities': {
                            'tools_count': len(tools_response.tools),
                            'vector_search': True,
                            'embedding_generation': True,
                            'cosine_similarity': True
                        },
                        'vector_statistics': stats,
                        'tools': [{'name': t.name, 'description': t.description} for t in tools_response.tools]
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _parse_stats(self, mcp_output: str) -> Dict[str, str]:
        """Parse vector statistics from MCP output"""
        stats = {}
        try:
            # Handle MCP format
            if hasattr(mcp_output, 'text'):
                actual_text = mcp_output.text
            elif "text='" in mcp_output:
                start = mcp_output.find("text='") + 6
                end = mcp_output.rfind("'")
                actual_text = mcp_output[start:end] if end > start else mcp_output
            else:
                actual_text = mcp_output
            
            lines = actual_text.replace('\\n', '\n').split('\n')
            in_stats = False
            
            for line in lines:
                line = line.strip()
                if line == 'VECTOR_STATS_START':
                    in_stats = True
                elif line == 'VECTOR_STATS_END':
                    in_stats = False
                elif in_stats and ':' in line:
                    key, value = line.split(':', 1)
                    stats[key] = value
                    
        except Exception as e:
            print(f"Stats parsing error: {e}")
        
        return stats

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
    """OCI GenAI Agent with vector search via MCP"""
    
    def __init__(self, agent_endpoint_id: str = None, compartment_id: str = None):
        self.config = load_config()
        
        # Use config values with parameter override capability
        self.agent_endpoint_id = agent_endpoint_id or self.config.oci.agent_endpoint_id
        self.compartment_id = compartment_id or self.config.oci.compartment_id
        self.agent = None
        
        self.instructions = """
        You are an expert IT support assistant powered by Oracle DB23ai vector search.
        
        Your capabilities use advanced vector embedding search:
        - search_knowledge_base_vector: Uses oci_cohere_rag_search_fixed function via MCP run-sql
        - get_vector_db_capabilities: Retrieves vector database statistics and capabilities
        
        Technical specifications:
        - Vector Function: oci_cohere_rag_search_fixed (PL/SQL function)
        - Embedding Model: OCI Cohere embed-english-v3.0 (1024 dimensions)
        - Distance Metric: COSINE similarity
        - Database: Oracle DB23ai with DBMS_VECTOR package
        - MCP Tool: run-sql for executing vector search function
        - Results: Ranked by similarity scores with detailed content
        
        When helping users:
        1. Use vector embedding search to find semantically similar content
        2. Provide answers based on similarity scores and rankings
        3. Include similarity scores to show confidence levels
        4. Explain that you're using vector embeddings for semantic search
        5. Reference specific content from the knowledge base
        6. Use the vector search function for all queries
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
                print("OCI GenAI Agent with DB23ai Vector Search setup complete")
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
    """Main function for vector DB23ai MCP RAG agent"""
    agent = VectorDB23aiMCPRAGAgent()
    agent.setup_agent()
    
    # Example usage
    print("Vector DB23ai MCP RAG Agent initialized")
    
if __name__ == "__main__":
    main()
