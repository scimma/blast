from pydantic_settings import BaseSettings
from pydantic import Field, AnyUrl, field_validator
from typing import Optional, List, Dict
import os

class Settings(BaseSettings):
    # General
    app_name: str = Field("AstroDash API", env="ASTRODASH_APP_NAME")
    environment: str = Field("production", env="ASTRODASH_ENVIRONMENT")
    debug: bool = Field(False, env="ASTRODASH_DEBUG")

    # API
    api_prefix: str = Field("/api/v1", env="ASTRODASH_API_PREFIX")
    allowed_hosts: List[str] = Field(["*"], env="ASTRODASH_ALLOWED_HOSTS")  # Allow all hosts for API usage
    cors_origins: List[str] = Field(["*"], env="ASTRODASH_CORS_ORIGINS")    # Allow all origins for API usage

    # Security Settings
    secret_key: str = Field("your-super-secret-key-here-make-it-very-long-and-secure-32-chars-min", env="ASTRODASH_SECRET_KEY")
    access_token_expire_minutes: int = Field(60 * 24, env="ASTRODASH_ACCESS_TOKEN_EXPIRE_MINUTES")

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(600, env="ASTRODASH_RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_burst_limit: int = Field(100, env="ASTRODASH_RATE_LIMIT_BURST_LIMIT")

    # Security Headers
    enable_hsts: bool = Field(True, env="ASTRODASH_ENABLE_HSTS")
    enable_csp: bool = Field(True, env="ASTRODASH_ENABLE_CSP")
    enable_permissions_policy: bool = Field(True, env="ASTRODASH_ENABLE_PERMISSIONS_POLICY")

    # Input Validation
    max_request_size: int = Field(100 * 1024 * 1024, env="ASTRODASH_MAX_REQUEST_SIZE")  # 100MB
    max_file_size: int = Field(50 * 1024 * 1024, env="ASTRODASH_MAX_FILE_SIZE")  # 50MB

    # Session Security
    session_cookie_secure: bool = Field(True, env="ASTRODASH_SESSION_COOKIE_SECURE")
    session_cookie_httponly: bool = Field(True, env="ASTRODASH_SESSION_COOKIE_HTTPONLY")
    session_cookie_samesite: str = Field("strict", env="ASTRODASH_SESSION_COOKIE_SAMESITE")

    # Database
    db_url: Optional[AnyUrl] = Field(None, env="ASTRODASH_DATABASE_URL")
    db_echo: bool = Field(False, env="ASTRODASH_DB_ECHO")

    # Data Storage (External to application code)
    data_dir: str = Field("/mnt/astrodash-data", env="ASTRODASH_DATA_DIR")
    storage_dir: str = Field("/mnt/astrodash-data", env="ASTRODASH_STORAGE_DIR")

    # ML Model Paths (External data directory)
    user_model_dir: str = Field("/mnt/astrodash-data/user_models", env="ASTRODASH_USER_MODEL_DIR")
    dash_model_path: str = Field("/mnt/astrodash-data/pre_trained_models/dash/pytorch_model.pth", env="ASTRODASH_DASH_MODEL_PATH")
    dash_training_params_path: str = Field("/mnt/astrodash-data/pre_trained_models/dash/training_params.pickle", env="ASTRODASH_DASH_TRAINING_PARAMS_PATH")
    transformer_model_path: str = Field("/mnt/astrodash-data/pre_trained_models/transformer/TF_wiserep_v6.pt", env="ASTRODASH_TRANSFORMER_MODEL_PATH")

    # Template and Line List Paths (External data directory)
    template_path: str = Field("/mnt/astrodash-data/pre_trained_models/templates/sn_and_host_templates.npz", env="ASTRODASH_TEMPLATE_PATH")
    line_list_path: str = Field("/mnt/astrodash-data/pre_trained_models/templates/sneLineList.txt", env="ASTRODASH_LINE_LIST_PATH")

    # ML Configuration Parameters
    # DASH model parameters
    nw: int = Field(1024, env="ASTRODASH_NW")  # Number of wavelength bins
    w0: float = Field(3500.0, env="ASTRODASH_W0")  # Minimum wavelength in Angstroms
    w1: float = Field(10000.0, env="ASTRODASH_W1")  # Maximum wavelength in Angstroms

    # Transformer model parameters
    label_mapping: Dict[str, int] = Field(
        {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4},
        env="ASTRODASH_LABEL_MAPPING"
    )

    # Transformer architecture parameters
    transformer_bottleneck_length: int = Field(1, env="ASTRODASH_TRANSFORMER_BOTTLENECK_LENGTH")
    transformer_model_dim: int = Field(128, env="ASTRODASH_TRANSFORMER_MODEL_DIM")
    transformer_num_heads: int = Field(4, env="ASTRODASH_TRANSFORMER_NUM_HEADS")
    transformer_num_layers: int = Field(6, env="ASTRODASH_TRANSFORMER_NUM_LAYERS")
    transformer_ff_dim: int = Field(256, env="ASTRODASH_TRANSFORMER_FF_DIM")
    transformer_dropout: float = Field(0.1, env="ASTRODASH_TRANSFORMER_DROPOUT")
    transformer_selfattn: bool = Field(False, env="ASTRODASH_TRANSFORMER_SELFATTN")

    # User model parameters
    user_model_reliability_threshold: float = Field(0.5, env="ASTRODASH_USER_MODEL_RELIABILITY_THRESHOLD")

    # Logging
    log_dir: str = Field("logs", env="ASTRODASH_LOG_DIR")
    log_level: str = Field("INFO", env="ASTRODASH_LOG_LEVEL")

    # Other
    osc_api_url: str = Field("https://api.astrocats.space", env="ASTRODASH_OSC_API_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from environment

    @field_validator("allowed_hosts", "cors_origins", mode="before")
    @classmethod
    def split_str(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @field_validator("label_mapping", mode="before")
    @classmethod
    def parse_label_mapping(cls, v):
        if isinstance(v, str):
            # Parse JSON string if provided as environment variable
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Fallback to default if parsing fails
                return {'Ia': 0, 'IIn': 1, 'SLSNe-I': 2, 'II': 3, 'Ib/c': 4}
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if v == "supersecret" and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("SECRET_KEY must be set to a secure value in production")
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed_environments = ["development", "staging", "production", "test"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v

    @field_validator("session_cookie_samesite")
    @classmethod
    def validate_session_cookie_samesite(cls, v):
        allowed_values = ["strict", "lax", "none"]
        if v not in allowed_values:
            raise ValueError(f"SESSION_COOKIE_SAMESITE must be one of: {allowed_values}")
        return v


def get_settings() -> Settings:
    return Settings()
