//! Dependency classification functions for different languages

use crate::DependencyType;

/// Classify Rust dependencies by their type
pub fn classify_rust_dependency(name: &str) -> DependencyType {
    match name {
        "tokio" | "async-std" | "actix-web" | "warp" | "axum" => DependencyType::WebFramework,
        "serde" | "serde_json" => DependencyType::Other("serialization".to_string()),
        "sqlx" | "diesel" | "rusqlite" => DependencyType::Database,
        "redis" | "bb8-redis" => DependencyType::Cache,
        "candle" | "tch" => DependencyType::MLFramework,
        "lapin" => DependencyType::MessageQueue,
        _ => DependencyType::Other("library".to_string()),
    }
}

/// Classify Node.js/JavaScript dependencies by their type
pub fn classify_node_dependency(name: &str) -> DependencyType {
    match name {
        "express" | "fastify" | "koa" | "hapi" => DependencyType::WebFramework,
        "react" | "vue" | "angular" => DependencyType::WebFramework,
        "mongoose" | "sequelize" | "typeorm" | "prisma" => DependencyType::Database,
        "redis" | "ioredis" | "memcached" => DependencyType::Cache,
        "tensorflow" | "torch" | "@tensorflow/tfjs" => DependencyType::MLFramework,
        "bull" | "agenda" => DependencyType::MessageQueue,
        _ => DependencyType::Other("library".to_string()),
    }
}

/// Classify Python dependencies by their type
pub fn classify_python_dependency(name: &str) -> DependencyType {
    match name {
        "django" | "flask" | "fastapi" | "tornado" | "starlette" => DependencyType::WebFramework,
        "sqlalchemy" | "django-orm" | "peewee" | "tortoise-orm" => DependencyType::Database,
        "redis" | "memcached" => DependencyType::Cache,
        "tensorflow" | "torch" | "pytorch" | "scikit-learn" | "keras" => {
            DependencyType::MLFramework
        }
        "celery" | "rq" => DependencyType::MessageQueue,
        _ => DependencyType::Other("library".to_string()),
    }
}

/// Classify Go dependencies by their type
pub fn classify_go_dependency(name: &str) -> DependencyType {
    if name.contains("gin") || name.contains("echo") || name.contains("fiber") {
        DependencyType::WebFramework
    } else if name.contains("gorm") || name.contains("database") || name.contains("sql") {
        DependencyType::Database
    } else if name.contains("redis") {
        DependencyType::Cache
    } else {
        DependencyType::Other("library".to_string())
    }
}

/// Classify Java dependencies by their type
pub fn classify_java_dependency(name: &str) -> DependencyType {
    match name {
        "spring-boot-starter-web" | "jersey" | "dropwizard" => DependencyType::WebFramework,
        "hibernate" | "jpa" | "mybatis" => DependencyType::Database,
        "jedis" | "lettuce" => DependencyType::Cache,
        "tensorflow" | "deeplearning4j" => DependencyType::MLFramework,
        "activemq" | "rabbitmq" => DependencyType::MessageQueue,
        _ => DependencyType::Other("library".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_dependency_classification() {
        assert!(matches!(
            classify_rust_dependency("tokio"),
            DependencyType::WebFramework
        ));
        assert!(matches!(
            classify_rust_dependency("sqlx"),
            DependencyType::Database
        ));
        assert!(matches!(
            classify_rust_dependency("redis"),
            DependencyType::Cache
        ));
        assert!(matches!(
            classify_rust_dependency("bb8-redis"),
            DependencyType::Cache
        ));
        assert!(matches!(
            classify_rust_dependency("candle"),
            DependencyType::MLFramework
        ));
        assert!(matches!(
            classify_rust_dependency("lapin"),
            DependencyType::MessageQueue
        ));
        assert!(matches!(
            classify_rust_dependency("unknown"),
            DependencyType::Other(_)
        ));
    }

    #[test]
    fn test_node_dependency_classification() {
        assert!(matches!(
            classify_node_dependency("express"),
            DependencyType::WebFramework
        ));
        assert!(matches!(
            classify_node_dependency("mongoose"),
            DependencyType::Database
        ));
        assert!(matches!(
            classify_node_dependency("redis"),
            DependencyType::Cache
        ));
        assert!(matches!(
            classify_node_dependency("ioredis"),
            DependencyType::Cache
        ));
        assert!(matches!(
            classify_node_dependency("tensorflow"),
            DependencyType::MLFramework
        ));
        assert!(matches!(
            classify_node_dependency("bull"),
            DependencyType::MessageQueue
        ));
    }

    #[test]
    fn test_python_dependency_classification() {
        assert!(matches!(
            classify_python_dependency("django"),
            DependencyType::WebFramework
        ));
        assert!(matches!(
            classify_python_dependency("sqlalchemy"),
            DependencyType::Database
        ));
        assert!(matches!(
            classify_python_dependency("redis"),
            DependencyType::Cache
        ));
        assert!(matches!(
            classify_python_dependency("tensorflow"),
            DependencyType::MLFramework
        ));
        assert!(matches!(
            classify_python_dependency("celery"),
            DependencyType::MessageQueue
        ));
    }

    #[test]
    fn test_go_dependency_classification() {
        assert!(matches!(
            classify_go_dependency("github.com/gin-gonic/gin"),
            DependencyType::WebFramework
        ));
        assert!(matches!(
            classify_go_dependency("gorm.io/gorm"),
            DependencyType::Database
        ));
        assert!(matches!(
            classify_go_dependency("github.com/go-redis/redis"),
            DependencyType::Cache
        ));
    }

    #[test]
    fn test_java_dependency_classification() {
        assert!(matches!(
            classify_java_dependency("spring-boot-starter-web"),
            DependencyType::WebFramework
        ));
        assert!(matches!(
            classify_java_dependency("hibernate"),
            DependencyType::Database
        ));
        assert!(matches!(
            classify_java_dependency("jedis"),
            DependencyType::Cache
        ));
        assert!(matches!(
            classify_java_dependency("tensorflow"),
            DependencyType::MLFramework
        ));
        assert!(matches!(
            classify_java_dependency("activemq"),
            DependencyType::MessageQueue
        ));
    }
}
