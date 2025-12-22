use thiserror::Error;

/// Errors that can occur in the authx crate
#[derive(Error, Debug)]
pub enum AuthError {
    #[error("certificate generation failed: {0}")]
    CertGeneration(String),

    #[error("certificate parsing failed: {0}")]
    CertParsing(String),

    #[error("certificate validation failed: {0}")]
    CertValidation(String),

    #[error("certificate expired: {0}")]
    CertExpired(String),

    #[error("invalid certificate signature: {0}")]
    InvalidSignature(String),

    #[error("hostname validation failed: {0}")]
    HostnameValidation(String),

    #[error("TLS configuration error: {0}")]
    TlsConfig(String),

    #[error("PEM encoding/decoding failed: {0}")]
    PemError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("rcgen error: {0}")]
    Rcgen(#[from] rcgen::RcgenError),

    #[error("rustls error: {0}")]
    Rustls(String),

    #[error("x509 parsing error: {0}")]
    X509(String),

    #[error("unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, AuthError>;

impl From<rustls::Error> for AuthError {
    fn from(e: rustls::Error) -> Self {
        AuthError::Rustls(e.to_string())
    }
}

impl From<x509_parser::error::X509Error> for AuthError {
    fn from(e: x509_parser::error::X509Error) -> Self {
        AuthError::X509(e.to_string())
    }
}

impl From<x509_parser::nom::Err<x509_parser::error::X509Error>> for AuthError {
    fn from(e: x509_parser::nom::Err<x509_parser::error::X509Error>) -> Self {
        AuthError::X509(e.to_string())
    }
}
