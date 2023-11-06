# codegenRAG
Example deployment of CodeGen using LlamaIndex RAG + Ray Serve

# Main Issues
- When do we activate the LLM?
- What do we pass in as the prompt?
- What prompts/few-shot prompts work well for multi-turn code generation?
- What relevant info can we give the LLM to improve outputs? (For example, giving a programming manual as RAG context)

Reach goals
- Can we do fast multi-turn prompting to generate even better outputs?
- If so, how do we know the LLM has "converged" onto the right answer?


#Done
[x] Checking if Ray Serve works
[x] Loading HuggingFace model
[x] General roadmap/skeleton code
[ ] Conditional activation of LLM based on user typing in VS code
[ ] Fleshing out RAG
[ ] Implementing streaming API call
