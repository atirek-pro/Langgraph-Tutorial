## What are Subgraphs in Langgraph?

A subgraph in Langgraph usually means a graph that is embedded and executed as a node inside another graph

## Benefits of Subgraphs

1. **Modularity**: Subgraphs allow you to break down complex graphs into smaller, manageable components. This makes it easier to understand, maintain, and reuse parts of your graph.
2. **Reusability**: Once a subgraph is created, it can be reused in multiple graphs, saving time and effort in graph construction.
3. **Maintainability**: Changes made to a subgraph will automatically propagate to all graphs that use it, ensuring consistency and reducing the risk of errors.

## Langgraph specific Benefits of Subgraphs

1. **Failure Isolation**: If a subgraph fails, it does not affect the execution of the parent graph. This allows for better error handling and debugging.
2. **State Separation**: Subgraphs can maintain their own state, which can be useful for managing complex workflows and data flows within a larger graph.
3. **Observability**: Subgraphs can be monitored and logged independently, providing better insights into the performance and behavior of different parts of the graph.

## Implementation of Subgraphs in Langgraph can be done in the following two ways:

1. **Invoke a graph from a node(ISOLATED STATES)** - subgraphs are calledfrom inside a node in the parent graph. This allows for dynamic execution of subgraphs based on the flow of the parent graph.
2. **Add a graph as a node(SHARED STATES)** - a subgraph is added as a node in the parent and shares the state keys with the parent. This allows for a more integrated approach where the subgraph can directly interact with the parent graph's state and data.
