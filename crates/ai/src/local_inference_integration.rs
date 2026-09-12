use crate::api_keys::ApiKeys;
use crate::local_inference::{LocalInferenceEndpoint, LocalInferenceSchema};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for local inference endpoints stored in settings
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct LocalInferenceConfig {
    /// Enabled local inference endpoints
    pub endpoints: Vec<LocalInferenceEndpoint>,
}

/// Extension trait for ApiKeys to support local inference
pub trait LocalInferenceSupport {
    fn has_local_endpoints(&self) -> bool;
    fn local_endpoints(&self) -> Vec<&LocalInferenceEndpoint>;
}

impl LocalInferenceSupport for ApiKeys {
    fn has_local_endpoints(&self) -> bool {
        !self.custom_endpoints.is_empty()
    }

    fn local_endpoints(&self) -> Vec<&crate::api_keys::CustomEndpoint> {
        self.custom_endpoints.iter().collect()
    }
}

/// Request builder for local inference endpoints
pub struct LocalInferenceRequest {
    pub endpoint_url: String,
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub api_key: Option<String>,
    pub schema: LocalInferenceSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl LocalInferenceRequest {
    pub fn new(
        endpoint_url: String,
        model: String,
        schema: LocalInferenceSchema,
    ) -> Self {
        Self {
            endpoint_url,
            model,
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            api_key: None,
            schema,
        }
    }

    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.messages.push(Message {
            role: role.into(),
            content: content.into(),
        });
    }

    /// Build the request payload based on schema
    pub fn build_payload(&self) -> Result<serde_json::Value, String> {
        match self.schema {
            LocalInferenceSchema::LmStudio | LocalInferenceSchema::ChatGptStyle => {
                self.build_openai_compatible_payload()
            }
            LocalInferenceSchema::AnthropicStyle => self.build_anthropic_compatible_payload(),
        }
    }

    fn build_openai_compatible_payload(&self) -> Result<serde_json::Value, String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": self.messages,
        });

        if let Some(temp) = self.temperature {
            payload["temperature"] = serde_json::json!(temp);
        }

        if let Some(tokens) = self.max_tokens {
            payload["max_tokens"] = serde_json::json!(tokens);
        }

        Ok(payload)
    }

    fn build_anthropic_compatible_payload(&self) -> Result<serde_json::Value, String> {
        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens.unwrap_or(1024),
        });

        if let Some(temp) = self.temperature {
            payload["temperature"] = serde_json::json!(temp);
        }

        Ok(payload)
    }

    /// Get the appropriate endpoint path based on schema
    pub fn get_endpoint_path(&self) -> &'static str {
        self.schema.default_path()
    }

    /// Build the full request URL
    pub fn build_url(&self) -> Result<String, String> {
        let base = self.endpoint_url.trim_end_matches('/');
        let path = self.get_endpoint_path();
        Ok(format!("{}{}", base, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_openai_compatible_payload() {
        let mut req = LocalInferenceRequest::new(
            "http://localhost:8000".to_string(),
            "mistral".to_string(),
            LocalInferenceSchema::LmStudio,
        );
        req.add_message("user", "Hello");
        req.temperature = Some(0.7);
        req.max_tokens = Some(256);

        let payload = req.build_payload().unwrap();
        assert_eq!(payload["model"], "mistral");
        assert_eq!(payload["temperature"], 0.7);
        assert_eq!(payload["max_tokens"], 256);
    }

    #[test]
    fn test_build_anthropic_compatible_payload() {
        let mut req = LocalInferenceRequest::new(
            "http://localhost:8000".to_string(),
            "claude".to_string(),
            LocalInferenceSchema::AnthropicStyle,
        );
        req.add_message("user", "Hello");
        req.temperature = Some(0.8);

        let payload = req.build_payload().unwrap();
        assert_eq!(payload["model"], "claude");
        assert_eq!(payload["temperature"], 0.8);
        assert!(payload["max_tokens"].is_number());
    }

    #[test]
    fn test_build_url() {
        let req = LocalInferenceRequest::new(
            "http://localhost:8000/".to_string(),
            "model".to_string(),
            LocalInferenceSchema::LmStudio,
        );
        let url = req.build_url().unwrap();
        assert_eq!(url, "http://localhost:8000/v1/chat/completions");
    }
}
