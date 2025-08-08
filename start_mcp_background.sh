#!/bin/bash
# Start DB23ai MCP server in background

export TNS_ADMIN=/home/ubuntu/wallet-db

echo "Starting DB23ai MCP Server in background..."
nohup ./sqlcl/bin/sql -mcp ADMIN/"OracleMCPServer@123"@genaidbmcp_high > mcp_server.log 2>&1 &
MCP_PID=$!
echo $MCP_PID > mcp_server.pid

echo " DB23ai MCP Server started with PID: $MCP_PID"
echo " Log file: mcp_server.log"
echo " PID file: mcp_server.pid"
echo " Check status: ps -p $MCP_PID"
