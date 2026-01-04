//! Parser for the .swarm DSL using pest

use crate::ast::*;
use crate::DslError;
use anyhow::Result;
use indexmap::IndexMap;
use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "swarm.pest"]
struct SwarmGrammar;

/// Parser for .swarm files
pub struct SwarmParser;

impl SwarmParser {
    /// Create a new parser instance
    pub fn new() -> Self {
        Self
    }

    /// Parse a .swarm file from a string
    pub fn parse(&self, input: &str) -> Result<SwarmFile, DslError> {
        let pairs = SwarmGrammar::parse(Rule::file, input).map_err(|e| {
            let (line, col) = match e.line_col {
                pest::error::LineColLocation::Pos((line, col)) => (line, col),
                pest::error::LineColLocation::Span((line, col), _) => (line, col),
            };
            DslError::ParseError {
                line,
                column: col,
                message: e.to_string(),
            }
        })?;

        let mut file = SwarmFile::new();

        for pair in pairs {
            if pair.as_rule() == Rule::file {
                for inner_pair in pair.into_inner() {
                    match inner_pair.as_rule() {
                        Rule::import_statement => {
                            file.imports.push(self.parse_import(inner_pair)?);
                        }
                        Rule::template_definition => {
                            file.templates.push(self.parse_template(inner_pair)?);
                        }
                        Rule::swarm_definition => {
                            file.swarms.push(self.parse_swarm_definition(inner_pair)?);
                        }
                        Rule::EOI => {}
                        _ => {}
                    }
                }
            }
        }

        Ok(file)
    }

    fn parse_import(&self, pair: Pair<Rule>) -> Result<Import, DslError> {
        let mut inner = pair.into_inner();
        let path = match self.parse_string(inner.next().unwrap())? {
            Value::String(s) => s,
            _ => {
                return Err(DslError::ParseError {
                    line: 0,
                    column: 0,
                    message: "Expected string for import path".to_string(),
                })
            }
        };
        let alias = inner.next().map(|p| p.as_str().to_string());

        Ok(Import { path, alias })
    }

    fn parse_template(&self, pair: Pair<Rule>) -> Result<Template, DslError> {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();

        let mut parameters = Vec::new();
        let mut body = None;

        for p in inner {
            match p.as_rule() {
                Rule::parameter_list => {
                    parameters = self.parse_parameter_list(p)?;
                }
                Rule::swarm_definition => {
                    body = Some(self.parse_swarm_definition(p)?);
                }
                _ => {}
            }
        }

        Ok(Template {
            name,
            parameters,
            body: body.ok_or_else(|| DslError::ParseError {
                line: 0,
                column: 0,
                message: "Template missing body".to_string(),
            })?,
        })
    }

    fn parse_swarm_definition(&self, pair: Pair<Rule>) -> Result<SwarmDefinition, DslError> {
        let mut inner = pair.into_inner();
        let name_pair = inner.next().unwrap();
        let name = match name_pair.as_rule() {
            Rule::identifier => name_pair.as_str().to_string(),
            Rule::interpolation => {
                // For now, just return the interpolation as-is
                // In a real implementation, we'd evaluate it
                name_pair.as_str().to_string()
            }
            _ => {
                return Err(DslError::ParseError {
                    line: 0,
                    column: 0,
                    message: "Expected identifier or interpolation for swarm name".to_string(),
                })
            }
        };

        let mut swarm = SwarmDefinition::new(name);

        for p in inner {
            match p.as_rule() {
                Rule::agents_block => {
                    swarm.agents = self.parse_agents_block(p)?;
                }
                Rule::connections_block => {
                    swarm.connections = self.parse_connections_block(p)?;
                }
                Rule::policies_block => {
                    swarm.policies = self.parse_policies_block(p)?;
                }
                Rule::functions_block => {
                    swarm.functions = self.parse_functions_block(p)?;
                }
                Rule::evolution_block => {
                    swarm.evolution = Some(self.parse_evolution_block(p)?);
                }
                Rule::affinity_block => {
                    swarm.affinity = Some(self.parse_affinity_block(p)?);
                }
                _ => {}
            }
        }

        Ok(swarm)
    }

    fn parse_agents_block(&self, pair: Pair<Rule>) -> Result<IndexMap<String, Agent>, DslError> {
        let mut agents = IndexMap::new();

        for p in pair.into_inner() {
            if p.as_rule() == Rule::agent_definition {
                let (name, agent) = self.parse_agent_definition(p)?;
                agents.insert(name, agent);
            }
        }

        Ok(agents)
    }

    fn parse_agent_definition(&self, pair: Pair<Rule>) -> Result<(String, Agent), DslError> {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();
        let agent_type = self.parse_agent_type(inner.next().unwrap())?;

        let mut properties = IndexMap::new();
        for p in inner {
            match p.as_rule() {
                Rule::agent_property => {
                    let (key, value) = self.parse_property(p)?;
                    properties.insert(key, value);
                }
                Rule::nested_block => {
                    let (key, value) = self.parse_nested_block(p)?;
                    properties.insert(key, value);
                }
                _ => {}
            }
        }

        Ok((
            name,
            Agent {
                agent_type: agent_type.0,
                type_params: agent_type.1,
                properties,
            },
        ))
    }

    fn parse_agent_type(&self, pair: Pair<Rule>) -> Result<(String, Vec<String>), DslError> {
        let mut inner = pair.into_inner();
        let type_name = inner.next().unwrap().as_str().to_string();

        let mut type_params = Vec::new();
        if let Some(params) = inner.next() {
            type_params = self.parse_type_params(params)?;
        }

        Ok((type_name, type_params))
    }

    fn parse_type_params(&self, pair: Pair<Rule>) -> Result<Vec<String>, DslError> {
        pair.into_inner()
            .map(|p| Ok(p.as_str().to_string()))
            .collect()
    }

    fn parse_connections_block(&self, pair: Pair<Rule>) -> Result<Vec<Connection>, DslError> {
        let mut connections = Vec::new();

        for p in pair.into_inner() {
            if p.as_rule() == Rule::connection_definition {
                connections.push(self.parse_connection_definition(p)?);
            }
        }

        Ok(connections)
    }

    fn parse_connection_definition(&self, pair: Pair<Rule>) -> Result<Connection, DslError> {
        let mut inner = pair.into_inner();
        let from = inner.next().unwrap().as_str().to_string();
        let to = inner.next().unwrap().as_str().to_string();

        let mut properties = IndexMap::new();
        for p in inner {
            if p.as_rule() == Rule::connection_property {
                let (key, value) = self.parse_property(p)?;
                properties.insert(key, value);
            }
        }

        Ok(Connection {
            from,
            to,
            properties,
        })
    }

    fn parse_policies_block(&self, pair: Pair<Rule>) -> Result<IndexMap<String, Value>, DslError> {
        let mut policies = IndexMap::new();

        for p in pair.into_inner() {
            if p.as_rule() == Rule::policy_definition {
                let (key, value) = self.parse_property(p)?;
                policies.insert(key, value);
            }
        }

        Ok(policies)
    }

    fn parse_functions_block(&self, pair: Pair<Rule>) -> Result<Vec<Function>, DslError> {
        let mut functions = Vec::new();

        for p in pair.into_inner() {
            if p.as_rule() == Rule::function_definition {
                functions.push(self.parse_function_definition(p)?);
            }
        }

        Ok(functions)
    }

    fn parse_function_definition(&self, pair: Pair<Rule>) -> Result<Function, DslError> {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();

        let mut parameters = Vec::new();
        let mut return_type = String::new();
        let mut body = Vec::new();

        for p in inner {
            match p.as_rule() {
                Rule::parameter_list => {
                    parameters = self.parse_parameter_list(p)?;
                }
                Rule::identifier => {
                    return_type = p.as_str().to_string();
                }
                Rule::statement => {
                    body.push(self.parse_statement(p)?);
                }
                _ => {}
            }
        }

        Ok(Function {
            name,
            parameters,
            return_type,
            body,
        })
    }

    fn parse_parameter_list(&self, pair: Pair<Rule>) -> Result<Vec<Parameter>, DslError> {
        pair.into_inner().map(|p| self.parse_parameter(p)).collect()
    }

    fn parse_parameter(&self, pair: Pair<Rule>) -> Result<Parameter, DslError> {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();
        let param_type = inner.next().unwrap().as_str().to_string();

        Ok(Parameter { name, param_type })
    }

    fn parse_statement(&self, pair: Pair<Rule>) -> Result<Statement, DslError> {
        let mut inner = pair.into_inner();
        let var_name = inner.next().unwrap().as_str().to_string();
        let expression = self.parse_expression(inner.next().unwrap())?;

        Ok(Statement {
            var_name,
            expression,
        })
    }

    fn parse_expression(&self, pair: Pair<Rule>) -> Result<Expression, DslError> {
        match pair.as_rule() {
            Rule::property_value => Ok(Expression::Value(self.parse_property_value(pair)?)),
            Rule::function_call => {
                let (name, args) = self.parse_function_call(pair)?;
                Ok(Expression::FunctionCall { name, args })
            }
            Rule::interpolation => {
                let inner = pair.into_inner().next().unwrap();
                Ok(Expression::Interpolation(Box::new(Expression::Value(
                    Value::Identifier(inner.as_str().to_string()),
                ))))
            }
            _ => Err(DslError::ParseError {
                line: 0,
                column: 0,
                message: format!("Unexpected expression type: {:?}", pair.as_rule()),
            }),
        }
    }

    fn parse_evolution_block(&self, pair: Pair<Rule>) -> Result<Evolution, DslError> {
        let mut properties = IndexMap::new();

        for p in pair.into_inner() {
            if p.as_rule() == Rule::evolution_property {
                let (key, value) = self.parse_property(p)?;
                properties.insert(key, value);
            }
        }

        Ok(Evolution { properties })
    }

    fn parse_affinity_block(&self, pair: Pair<Rule>) -> Result<Affinity, DslError> {
        let mut rules = IndexMap::new();

        for p in pair.into_inner() {
            if p.as_rule() == Rule::affinity_rule {
                let (key, value) = self.parse_property(p)?;
                rules.insert(key, value);
            }
        }

        Ok(Affinity { rules })
    }

    fn parse_property(&self, pair: Pair<Rule>) -> Result<(String, Value), DslError> {
        let mut inner = pair.into_inner();
        let key = inner.next().unwrap().as_str().to_string();
        let value = self.parse_property_value(inner.next().unwrap())?;

        Ok((key, value))
    }

    fn parse_property_value(&self, pair: Pair<Rule>) -> Result<Value, DslError> {
        let inner = pair.into_inner().next().unwrap();

        match inner.as_rule() {
            Rule::string => self.parse_string(inner),
            Rule::number => self.parse_number(inner),
            Rule::boolean => Ok(Value::Boolean(inner.as_str() == "true")),
            Rule::identifier => Ok(Value::Identifier(inner.as_str().to_string())),
            Rule::array => self.parse_array(inner),
            Rule::object => self.parse_object(inner),
            Rule::range => self.parse_range(inner),
            Rule::function_call => {
                let (name, args) = self.parse_function_call(inner)?;
                Ok(Value::FunctionCall { name, args })
            }
            Rule::tier_preference => self.parse_tier_preference(inner),
            _ => Err(DslError::ParseError {
                line: 0,
                column: 0,
                message: format!("Unexpected property value type: {:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_string(&self, pair: Pair<Rule>) -> Result<Value, DslError> {
        let s = pair.as_str();
        // Remove quotes
        let s = &s[1..s.len() - 1];
        Ok(Value::String(s.to_string()))
    }

    fn parse_number(&self, pair: Pair<Rule>) -> Result<Value, DslError> {
        let num = pair
            .as_str()
            .parse::<f64>()
            .map_err(|_| DslError::ParseError {
                line: 0,
                column: 0,
                message: format!("Invalid number: {}", pair.as_str()),
            })?;
        Ok(Value::Number(num))
    }

    fn parse_array(&self, pair: Pair<Rule>) -> Result<Value, DslError> {
        let values: Result<Vec<_>, _> = pair
            .into_inner()
            .map(|p| self.parse_property_value(p))
            .collect();
        Ok(Value::Array(values?))
    }

    fn parse_object(&self, pair: Pair<Rule>) -> Result<Value, DslError> {
        let mut map = IndexMap::new();

        for p in pair.into_inner() {
            if p.as_rule() == Rule::object_field {
                let mut inner = p.into_inner();
                let key = inner.next().unwrap().as_str().to_string();
                let value = self.parse_property_value(inner.next().unwrap())?;
                map.insert(key, value);
            }
        }

        Ok(Value::Object(map))
    }

    fn parse_range(&self, pair: Pair<Rule>) -> Result<Value, DslError> {
        let mut inner = pair.into_inner();
        let start =
            inner
                .next()
                .unwrap()
                .as_str()
                .parse::<f64>()
                .map_err(|_| DslError::ParseError {
                    line: 0,
                    column: 0,
                    message: "Invalid range start".to_string(),
                })?;
        let end =
            inner
                .next()
                .unwrap()
                .as_str()
                .parse::<f64>()
                .map_err(|_| DslError::ParseError {
                    line: 0,
                    column: 0,
                    message: "Invalid range end".to_string(),
                })?;

        Ok(Value::Range { start, end })
    }

    fn parse_function_call(&self, pair: Pair<Rule>) -> Result<(String, Vec<Value>), DslError> {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();

        let mut args = Vec::new();
        for p in inner {
            args.push(self.parse_property_value(p)?);
        }

        Ok((name, args))
    }

    fn parse_tier_preference(&self, pair: Pair<Rule>) -> Result<Value, DslError> {
        let tiers: Result<Vec<_>, _> = pair.into_inner().map(|p| self.parse_tier_type(p)).collect();
        Ok(Value::Array(tiers?))
    }

    fn parse_tier_type(&self, pair: Pair<Rule>) -> Result<Value, DslError> {
        let tier = match pair.as_str() {
            "GPU" => TierType::GPU,
            "CPU" => TierType::CPU,
            "NVMe" => TierType::NVMe,
            "Memory" => TierType::Memory,
            _ => {
                return Err(DslError::ParseError {
                    line: 0,
                    column: 0,
                    message: format!("Unknown tier type: {}", pair.as_str()),
                })
            }
        };
        Ok(Value::TierType(tier))
    }

    fn parse_nested_block(&self, pair: Pair<Rule>) -> Result<(String, Value), DslError> {
        let mut inner = pair.into_inner();
        let key = inner.next().unwrap().as_str().to_string();

        let mut map = IndexMap::new();
        for p in inner {
            if p.as_rule() == Rule::object_field {
                let mut field_inner = p.into_inner();
                let field_key = field_inner.next().unwrap().as_str().to_string();
                let field_value = self.parse_property_value(field_inner.next().unwrap())?;
                map.insert(field_key, field_value);
            }
        }

        Ok((key, Value::Object(map)))
    }
}

impl Default for SwarmParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_swarm() {
        let parser = SwarmParser::new();
        let input = "swarm test {}";
        let result = parser.parse(input);
        assert!(result.is_ok());

        let file = result.unwrap();
        assert_eq!(file.swarms.len(), 1);
        assert_eq!(file.swarms[0].name, "test");
    }

    #[test]
    fn test_parse_agent() {
        let parser = SwarmParser::new();
        let input = r#"
            swarm test {
                agents {
                    web: WebAgent {
                        replicas: 3,
                        memory: "4Gi",
                    }
                }
            }
        "#;

        let result = parser.parse(input);
        assert!(result.is_ok());

        let file = result.unwrap();
        let swarm = &file.swarms[0];
        assert!(swarm.agents.contains_key("web"));

        let agent = &swarm.agents["web"];
        assert_eq!(agent.agent_type, "WebAgent");
        assert_eq!(agent.properties["replicas"], Value::Number(3.0));
        assert_eq!(agent.properties["memory"], Value::String("4Gi".to_string()));
    }

    #[test]
    fn test_parse_connection() {
        let parser = SwarmParser::new();
        let input = r#"
            swarm test {
                connections {
                    frontend -> backend: {
                        protocol: "grpc",
                        retry: true,
                    }
                }
            }
        "#;

        let result = parser.parse(input);
        assert!(result.is_ok());

        let file = result.unwrap();
        let swarm = &file.swarms[0];
        assert_eq!(swarm.connections.len(), 1);

        let conn = &swarm.connections[0];
        assert_eq!(conn.from, "frontend");
        assert_eq!(conn.to, "backend");
        assert_eq!(
            conn.properties["protocol"],
            Value::String("grpc".to_string())
        );
        assert_eq!(conn.properties["retry"], Value::Boolean(true));
    }

    #[test]
    fn test_parse_range() {
        let parser = SwarmParser::new();
        let input = r#"
            swarm test {
                agents {
                    web: WebAgent {
                        replicas: 3..10,
                    }
                }
            }
        "#;

        let result = parser.parse(input);
        assert!(result.is_ok());

        let file = result.unwrap();
        let agent = &file.swarms[0].agents["web"];

        match &agent.properties["replicas"] {
            Value::Range { start, end } => {
                assert_eq!(*start, 3.0);
                assert_eq!(*end, 10.0);
            }
            _ => panic!("Expected range value"),
        }
    }

    #[test]
    fn test_parse_array() {
        let parser = SwarmParser::new();
        let input = r#"
            swarm test {
                agents {
                    web: WebAgent {
                        ports: [80, 443, 8080],
                    }
                }
            }
        "#;

        let result = parser.parse(input);
        assert!(result.is_ok());

        let file = result.unwrap();
        let agent = &file.swarms[0].agents["web"];

        match &agent.properties["ports"] {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], Value::Number(80.0));
                assert_eq!(arr[1], Value::Number(443.0));
                assert_eq!(arr[2], Value::Number(8080.0));
            }
            _ => panic!("Expected array value"),
        }
    }

    #[test]
    fn test_parse_object() {
        let parser = SwarmParser::new();
        let input = r#"
            swarm test {
                agents {
                    web: WebAgent {
                        resources: {
                            cpu: 2,
                            memory: "4Gi",
                            gpu: 0.5
                        }
                    }
                }
            }
        "#;

        let result = parser.parse(input);
        if let Err(e) = &result {
            println!("Parse error: {:?}", e);
        }
        assert!(result.is_ok());

        let file = result.unwrap();
        let agent = &file.swarms[0].agents["web"];

        match &agent.properties["resources"] {
            Value::Object(obj) => {
                assert_eq!(obj["cpu"], Value::Number(2.0));
                assert_eq!(obj["memory"], Value::String("4Gi".to_string()));
                assert_eq!(obj["gpu"], Value::Number(0.5));
            }
            _ => panic!("Expected object value"),
        }
    }
}
