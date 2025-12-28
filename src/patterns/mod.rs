pub mod ai;
pub mod analyzer;
pub mod compression;
pub mod database;
pub mod vectors;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DetectionResult {
    pub pattern: String,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompressionDetectionResult {
    pub pattern: String,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorDetectionResult {
    pub pattern: String,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DatabaseDetectionResult {
    pub pattern: String,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AIDetectionResult {
    pub pattern: String,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PatternProfile {
    pub compression: CompressionDetectionResult,
    pub vector: VectorDetectionResult,
    pub database: DatabaseDetectionResult,
    pub ai: AIDetectionResult,
    pub primary_pattern: String,
    pub optimization_suggestion: String,
}
