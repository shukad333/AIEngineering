from mcp.server.mcpserver import MCPServer

mcp = MCPServer("qatar-docs")
_rag = None

def rag():
    # Lazy-load: bge-m3 takes a while to load, and clients time out slow-starting servers
    global _rag
    if _rag is None:
        import arabic_rag
        _rag = arabic_rag
    return _rag

@mcp.tool()
def search_docs(query: str, k: int = 4) -> str:
    """Search the Arabic/English document collection and return the most relevant
    passages with their source file and a similarity score. Use this for any
    question about the contents of the documents. Works with Arabic or English queries."""
    r = rag()
    idx, scores = r.retrieve(query, min(max(k, 1), 10))
    return "\n\n".join(
        f"[{r.meta[i]['file']}#{r.meta[i]['chunk']}] (score {s:.2f})\n{r.chunks[i]}"
        for i, s in zip(idx, scores))

@mcp.tool()
def list_documents() -> list[str]:
    """List the document files available to search."""
    return sorted({m["file"] for m in rag().meta})

if __name__ == "__main__":
    mcp.run()          # stdio transport