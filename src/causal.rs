use crate::equation::Expression;
use crate::error::{LmmError, Result};
use crate::traits::Causal;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct CausalNode {
    pub id: usize,
    pub name: String,
    pub equation: Option<Expression>,
    pub observed_value: Option<f64>,
    pub intervened_value: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CausalGraph {
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<CausalEdge>,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, name: &str, equation: Option<Expression>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(CausalNode {
            id,
            name: name.to_string(),
            equation,
            observed_value: None,
            intervened_value: None,
        });
        id
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) -> Result<()> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return Err(LmmError::CausalError(format!(
                "Node index out of bounds: from={from}, to={to}, len={}",
                self.nodes.len()
            )));
        }
        self.edges.push(CausalEdge { from, to, weight });
        Ok(())
    }

    pub fn parents(&self, node_id: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.to == node_id)
            .map(|e| e.from)
            .collect()
    }

    pub fn children(&self, node_id: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.from == node_id)
            .map(|e| e.to)
            .collect()
    }

    pub fn markov_blanket(&self, node_id: usize) -> Vec<usize> {
        let mut blanket = HashSet::new();
        for p in self.parents(node_id) {
            blanket.insert(p);
        }
        for c in self.children(node_id) {
            blanket.insert(c);
            for co_parent in self.parents(c) {
                if co_parent != node_id {
                    blanket.insert(co_parent);
                }
            }
        }
        let mut result: Vec<usize> = blanket.into_iter().collect();
        result.sort();
        result
    }

    pub fn has_cycle(&self) -> bool {
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut rec_stack = vec![false; n];

        fn dfs(
            node: usize,
            edges: &[CausalEdge],
            visited: &mut Vec<bool>,
            rec_stack: &mut Vec<bool>,
        ) -> bool {
            visited[node] = true;
            rec_stack[node] = true;
            for edge in edges.iter().filter(|e| e.from == node) {
                if !visited[edge.to] {
                    if dfs(edge.to, edges, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack[edge.to] {
                    return true;
                }
            }
            rec_stack[node] = false;
            false
        }

        for i in 0..n {
            if !visited[i] && dfs(i, &self.edges, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        false
    }

    pub fn topological_order(&self) -> Result<Vec<usize>> {
        if self.has_cycle() {
            return Err(LmmError::CausalError("Graph contains a cycle".into()));
        }
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        for e in &self.edges {
            in_degree[e.to] += 1;
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::new();
        while !queue.is_empty() {
            queue.sort();
            let node = queue.remove(0);
            order.push(node);
            for e in self.edges.iter().filter(|e| e.from == node) {
                in_degree[e.to] -= 1;
                if in_degree[e.to] == 0 {
                    queue.push(e.to);
                }
            }
        }
        Ok(order)
    }

    pub fn forward_pass(&mut self) -> Result<HashMap<usize, f64>> {
        let order = self.topological_order()?;
        let mut values: HashMap<usize, f64> = HashMap::new();

        for id in order {
            let node = &self.nodes[id];
            if let Some(v) = node.intervened_value {
                values.insert(id, v);
                continue;
            }
            if let Some(v) = node.observed_value {
                values.insert(id, v);
                continue;
            }
            if let Some(eq) = &node.equation {
                let mut bindings = HashMap::new();
                for parent_id in self.parents(id) {
                    let parent_name = self.nodes[parent_id].name.clone();
                    if let Some(&pv) = values.get(&parent_id) {
                        bindings.insert(parent_name, pv);
                    }
                }
                let v = eq.evaluate(&bindings).unwrap_or(0.0);
                values.insert(id, v);
            }
        }
        Ok(values)
    }

    pub fn counterfactual(
        &self,
        observed: &HashMap<usize, f64>,
        intervention: &HashMap<usize, f64>,
    ) -> Result<HashMap<usize, f64>> {
        let mut twin = self.clone();
        for (&id, &v) in observed {
            twin.nodes[id].observed_value = Some(v);
        }
        for (&id, &v) in intervention {
            twin.nodes[id].intervened_value = Some(v);
        }
        twin.forward_pass()
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Causal for CausalGraph {
    fn intervene(&mut self, var: &str, value: f64) -> Result<()> {
        let node = self
            .nodes
            .iter_mut()
            .find(|n| n.name == var)
            .ok_or_else(|| LmmError::CausalError(format!("Node '{}' not found", var)))?;
        node.intervened_value = Some(value);
        Ok(())
    }
}
