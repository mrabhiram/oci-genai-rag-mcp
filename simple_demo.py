#!/usr/bin/env python3
import os
from oci.addons.adk import Agent, AgentClient
from vector_sync_tools import search_knowledge_base_vector, get_vector_db_capabilities

# Configuration
agent_endpoint_id = 'ocid1.genaiagentendpoint.oc1.iad.amaaaaaaieosruaaxpzdp7gwdk3hd5selesyszveooa3huxi3r2y2m3tsjkq'
region = 'us-ashburn-1'

# Create client first with region
client = AgentClient(
    auth_type="api_key",
    region=region
)

# Create agent with client
agent = Agent(
    client=client,
    agent_endpoint_id=agent_endpoint_id,
    tools=[search_knowledge_base_vector, get_vector_db_capabilities]
)

print("\nOracle GenAI + DB23ai Vector Search Demo")
print("=" * 50)
print("Type 'quit' to exit\n")

# Interactive loop
while True:
    query = input("You: ").strip()
    if query.lower() in ['quit', 'exit']:
        break
    if query:
        try:
            response = agent.run(query)
            print(f"\nAgent: {response.get('content', 'No response')}\n")
        except Exception as e:
            print(f"Error: {e}\n")
