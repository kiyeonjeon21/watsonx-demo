# Multi-Agent Collaboration Example

This example demonstrates multiple agents working together to provide comprehensive information about WXO and WX Assistant products.

## Setup
1. Import tools and agents: `./import-all.sh`
2. Select `IBM_Product_Specialist`

## Usage Examples
- "Tell me about WXO features"
- "What are Watson Assistant pricing options?"
- "Compare WXO and Watson Assistant"
- "What can you tell me about Cognos Analytics?"
- "Which product is better for my analytics needs?"
- "Please compare cognos, orchestrate, and assistant"

## Agents
- **IBM Product Specialist**: Main agent that coordinates with other specialists
- **Wx_Orchestrate**: Provides WXO product information
- **Wx_Assistant**: Provides Watson Assistant product information  
- **Cognos_Analytics**: Provides Cognos Analytics product information

## Cleanup
Run `./remove-all.sh` to remove all imported tools and agents.
