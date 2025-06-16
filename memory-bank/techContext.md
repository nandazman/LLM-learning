# Technical Context

## Technology Stack

### Backend
- **FastAPI**: Main API framework
- **LangChain/LangGraph**: AI/LLM orchestration
- **Neo4j**: Graph database
- **Vector Store**: Document embeddings and search
- **Docker**: Containerization
- **Pytest**: Testing framework

### Integration
- **Strapi CMS**: Forum platform
- **LLM Providers**:
  - OpenAI
  - Anthropic
  - (Extensible for other providers)

## Development Setup

### Environment Variables
```bash
STRAPI_AGENT_KEY=    # API key for Strapi integration
OPENAI_API_KEY=      # OpenAI API credentials
ANTHROPIC_API_KEY=   # Anthropic API credentials
NEO4J_URI=          # Neo4j connection string
NEO4J_USER=         # Neo4j username
NEO4J_PASS=         # Neo4j password
VECTOR_DB_URL=      # Vector store endpoint
```

### Project Structure
```
ai_service/
├─ main.py               # FastAPI entry-point
├─ api/
│  └─ routes.py          # API endpoints
├─ agents/               # Agent implementations
├─ llm/                  # LLM handling
├─ data/                 # Data storage
├─ utils/                # Utilities
├─ tests/                # Test suite
├─ Dockerfile
└─ docker-compose.yml
```

## Technical Constraints

### Performance
- Response time targets for thread answers
- Rate limiting for API endpoints
- Vector store query optimization
- Neo4j query performance

### Security
- API key validation
- Domain whitelisting
- Rate limiting
- Secure credential management

### Scalability
- Docker-based deployment
- Stateless API design
- Database connection pooling
- Caching strategies

## Dependencies

### Python Packages
- fastapi
- langchain
- langgraph
- neo4j
- pytest
- docker
- (Additional dependencies in requirements.txt)

### External Services
- Strapi CMS
- LLM Providers
- Neo4j Database
- Vector Store

## Development Workflow
1. Local development with Docker Compose
2. Test suite execution
3. API documentation generation
4. Deployment preparation 