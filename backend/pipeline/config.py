"""Pipeline configuration loader (plug-and-play YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

@dataclass
class ValidationConfig:
    enabled: bool = True
    allowed_formats: List[str] = field(default_factory=list)
    duration_min: Optional[float] = None
    duration_max: Optional[float] = None
    size_max_mb: Optional[float] = None
    sample_rate_allowed: List[int] = field(default_factory=list)
    download_timeout: float = 60.0
    retry_count: int = 2
    retry_delay: float = 2.0
    rejection_webhook_url: Optional[str] = None
    rejection_webhook_secret: Optional[str] = None


@dataclass
class ClearAudioConfig:
    target_format: str = "wav"
    target_sample_rate: int = 16000
    normalization_enabled: bool = True
    normalization_mode: str = "loudnorm"  # loudnorm | dynaudnorm
    normalization_level_db: float = -23.0
    denoise_enabled: bool = True
    denoise_intensity: str = "medium"
    trim_enabled: bool = True
    trim_silence_threshold: float = -40.0
    trim_min_silence_duration: float = 0.3
    chunking_enabled: bool = True
    chunking_max_duration_sec: float = 30.0
    chunking_overlap_sec: float = 2.0
    metadata_include_fields: List[str] = field(default_factory=lambda: ["duration", "size", "snr_db"])


@dataclass
class PreprocessingConfig:
    csv_path: str
    s3_prefix: str
    add_timestamp: bool = False
    version: Optional[str] = None
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    # When clear_audio is None, heavy audio processing (FFmpeg/etc.) is skipped.
    clear_audio: Optional[ClearAudioConfig] = None
    store_processed_artifacts: bool = False
    store_ml_ready_artifacts: bool = False


@dataclass
class LabelSchemaConfig:
    fields: List[str] = field(default_factory=lambda: ["transcription"])


@dataclass
class PreAnnotationConfig:
    enabled: bool = True
    label_schema: LabelSchemaConfig = field(default_factory=LabelSchemaConfig)


@dataclass
class ASRConfig:
    model: str = "assemblyai"
    language: Optional[str] = None
    task: str = "transcribe"


@dataclass
class TimestampConfig:
    format: str = "s"


@dataclass
class OutputConfig:
    format: str = "json"


@dataclass
class TranscriptionConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    timestamp: TimestampConfig = field(default_factory=TimestampConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sentiment_enabled: bool = False


@dataclass
class KeywordsConfig:
    enabled: bool = False
    model: str = "default"


@dataclass
class TaggingConfig:
    enabled: bool = False
    labels: List[str] = field(default_factory=list)


@dataclass
class AdditionalAnnotationConfig:
    keywords: KeywordsConfig = field(default_factory=KeywordsConfig)
    tagging: TaggingConfig = field(default_factory=TaggingConfig)


@dataclass
class TemplateMappingRule:
    source: str
    target: str


@dataclass
class MLValidationConfig:
    """Optional ML-based validation configuration."""

    enabled: bool = False
    # Free-form rules/thresholds for future use
    min_confidence: Optional[float] = None
    max_duration_diff_sec: Optional[float] = None
    rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationTemplateConfig:
    template_id: str = "default-template"
    version: str = "1.0"
    mapping_rules: List[TemplateMappingRule] = field(default_factory=list)


@dataclass
class AnnotationModuleConfig:
    label_config: str
    preannotation: Optional[PreAnnotationConfig] = None
    transcription: Optional[TranscriptionConfig] = None
    enrichment: Optional[AdditionalAnnotationConfig] = None
    ml_validation: Optional[MLValidationConfig] = None
    template: AnnotationTemplateConfig = field(default_factory=AnnotationTemplateConfig)


@dataclass
class PostProcessingConfig:
    final_csv_path: str = "./.tmp/final_report.csv"
    s3_prefix: Optional[str] = None
    extract_audio_segments: bool = True
    # Post-processing options
    post_merge_strategy: str = "prefer-human"
    post_schema_enabled: bool = False
    post_schema_id: Optional[str] = None
    post_output_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    post_s3_bucket: Optional[str] = None
    post_s3_prefix: Optional[str] = None
    completion_webhook_url: Optional[str] = None
    completion_webhook_secret: Optional[str] = None
    # Output fields configuration - specify which fields from Label Studio to include in final JSON
    # Default includes standard fields. Add any custom fields from your Label Studio template here.
    output_fields: List[str] = field(default_factory=lambda: ["start", "end", "speaker", "text", "language", "end_of_speech"])


@dataclass
class MLBackendConfig:
    enabled: bool = True
    url: Optional[str] = None
    token: Optional[str] = None


@dataclass
class LSExportConfig:
    format: str = "json"


@dataclass
class StorageConfig:
    enabled: bool = False
    storage_type: str = "s3"  # s3, gcs, azure
    bucket: Optional[str] = None
    prefix: Optional[str] = None  # S3 prefix/folder path
    region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    use_blob_urls: bool = True  # Whether to use direct blob URLs for tasks
    title: Optional[str] = None  # Storage connection title
    # Source storage: for importing tasks (reading audio files)
    source_enabled: bool = True
    source_prefix: Optional[str] = None  # Override prefix for source storage
    # Target storage: for exporting annotations
    target_enabled: bool = False
    target_prefix: Optional[str] = None  # Prefix for target storage (e.g., annotations/)


@dataclass
class WebhookConfig:
    enabled: bool = False
    url: Optional[str] = None
    events: List[str] = field(default_factory=lambda: [
        "TASK_CREATED",
        "ANNOTATION_CREATED",
        "ANNOTATION_UPDATED",
        "TASK_COMPLETED",
    ])
    is_active: bool = True


@dataclass
class LabelStudioProjectConfig:
    project_name_prefix: str = "AUDIO_"
    description: str = ""
    template_id: Optional[str] = None
    batch_size: int = 50
    ml_backend: MLBackendConfig = field(default_factory=MLBackendConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    run_pre_annotate: bool = True
    reviewer_ids: List[str] = field(default_factory=list)
    export: LSExportConfig = field(default_factory=LSExportConfig)


@dataclass
class RejectionWebhookConfig:
    url: Optional[str] = None
    secret: Optional[str] = None


@dataclass
class RejectionLambdaConfig:
    function_name: Optional[str] = None
    payload_fields: List[str] = field(default_factory=list)


@dataclass
class RejectionLoggingConfig:
    destination: Optional[str] = None
    fields: List[str] = field(default_factory=list)


@dataclass
class RejectionHandlerConfig:
    webhook: RejectionWebhookConfig = field(default_factory=RejectionWebhookConfig)
    lambda_config: RejectionLambdaConfig = field(default_factory=RejectionLambdaConfig)
    logging: RejectionLoggingConfig = field(default_factory=RejectionLoggingConfig)


@dataclass
class WorkflowStepConfig:
    enabled: bool = True
    description: Optional[str] = None


@dataclass
class WorkflowConfig:
    steps: List[str] = field(default_factory=list)
    preprocessing: WorkflowStepConfig = field(default_factory=WorkflowStepConfig)
    annotation: WorkflowStepConfig = field(default_factory=WorkflowStepConfig)
    postprocessing: WorkflowStepConfig = field(default_factory=WorkflowStepConfig)


@dataclass
class PipelineConfig:
    client_id: str
    preprocessing: PreprocessingConfig
    annotation: AnnotationModuleConfig
    postprocessing: Optional[PostProcessingConfig]
    ls_project: LabelStudioProjectConfig
    rejection_handler: Optional[RejectionHandlerConfig]
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls._from_raw(raw)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls._from_raw(data)

    @classmethod
    def _from_raw(cls, data: Dict[str, Any]) -> "PipelineConfig":
        # Support both old flat structure and new clientConfig/devConfig split
        if "clientConfig" in data:
            client_config = data.get("clientConfig", {})
            dev_config = data.get("devConfig", {})
            client_id = client_config.get("clientId", "default")
            workflow = cls._build_workflow(client_config.get("workflow") or {})

            # Build preprocessing from clientConfig.preprocessing
            preprocessing = cls._build_preprocessing(
                client_config.get("preprocessing") or {},
                dev_config=dev_config,
                workflow_enabled=workflow.preprocessing.enabled,
            )
            annotation = cls._build_annotation(
                client_config.get("annotation") or {},
                workflow_enabled=workflow.annotation.enabled,
            )
            postprocessing = cls._build_postprocessing(
                client_config.get("postProcessing") or {},
                dev_config=dev_config,
                workflow_enabled=workflow.postprocessing.enabled,
            )
            ls_project = cls._build_ls_project(dev_config)
            rejection_handler = cls._build_rejection(dev_config)
        else:
            # Legacy flat structure
            client_id = data.get("clientId", "default")
            workflow = WorkflowConfig()
            preprocessing = cls._build_preprocessing(
                data.get("preprocessing") or {},
                workflow_enabled=workflow.preprocessing.enabled,
            )
            annotation = cls._build_annotation(
                data.get("annotation") or {},
                workflow_enabled=workflow.annotation.enabled,
            )
            postprocessing = cls._build_postprocessing(
                data.get("postProcessing") or data.get("postprocessing") or data.get("delivery"),
                workflow_enabled=workflow.postprocessing.enabled,
            )
            ls_project = cls._build_ls_project(data)
            rejection_handler = cls._build_rejection(data)

        return cls(
            client_id=client_id,
            preprocessing=preprocessing,
            annotation=annotation,
            postprocessing=postprocessing,
            ls_project=ls_project,
            rejection_handler=rejection_handler,
            workflow=workflow,
        )

    @staticmethod
    def _build_workflow(raw: Dict[str, Any]) -> WorkflowConfig:
        if not raw:
            return WorkflowConfig()

        def _step(name: str) -> WorkflowStepConfig:
            step_raw = raw.get(name) or {}
            return WorkflowStepConfig(
                enabled=step_raw.get("enabled", True),
                description=step_raw.get("description"),
            )

        return WorkflowConfig(
            steps=raw.get("steps") or [],
            preprocessing=_step("preprocessing"),
            annotation=_step("annotation"),
            postprocessing=_step("postProcessing"),
        )

    @staticmethod
    def _build_preprocessing(
        raw: Dict[str, Any],
        dev_config: Optional[Dict[str, Any]] = None,
        workflow_enabled: bool = True,
    ) -> PreprocessingConfig:
        if not raw:
            raise ValueError("preprocessing config is required")
        data = raw.copy()
        validation_raw = data.get("validation") or {}
        duration = validation_raw.pop("duration", {})
        size = validation_raw.pop("size", {})
        sample = validation_raw.pop("sampleRate", {})
        download = validation_raw.pop("download", {})
        retry = download.get("retry", {})
        webhook = validation_raw.pop("rejectionWebhook", {})
        validation = ValidationConfig()
        validation.enabled = validation_raw.get("enabled", True)
        validation.allowed_formats = validation_raw.get("allowedFormats") or []
        validation.duration_min = duration.get("minSec") or duration.get("min")
        validation.duration_max = duration.get("maxSec") or duration.get("max")
        validation.size_max_mb = size.get("maxMB") or size.get("max")
        validation.sample_rate_allowed = sample.get("allowed") or []
        validation.download_timeout = download.get("timeoutSec") or download.get("timeout") or validation.download_timeout
        validation.retry_count = download.get("retryCount") or retry.get("count") or validation.retry_count
        validation.retry_delay = download.get("retryDelaySec") or retry.get("delay") or validation.retry_delay
        validation.rejection_webhook_url = webhook.get("url")
        validation.rejection_webhook_secret = webhook.get("secret")

        clear_raw = data.get("clearAudio")
        clear: Optional[ClearAudioConfig]
        if clear_raw:
            # Optional explicit flag: clearAudio.enabled: false → skip heavy audio processing
            if clear_raw.get("enabled") is False:
                clear = None
            else:
                clear_raw = clear_raw or {}
                clear = ClearAudioConfig(
                    target_format=(clear_raw.get("target") or {}).get("format", "wav"),
                    target_sample_rate=(clear_raw.get("target") or {}).get("sampleRate", 16000),
                    normalization_enabled=(clear_raw.get("normalization") or {}).get("enabled", True),
                    normalization_mode=(clear_raw.get("normalization") or {}).get("mode", "loudnorm"),
                    normalization_level_db=(clear_raw.get("normalization") or {}).get("levelDB", -23.0),
                    denoise_enabled=(clear_raw.get("denoise") or {}).get("enabled", True),
                    denoise_intensity=(clear_raw.get("denoise") or {}).get("intensity", "medium"),
                    trim_enabled=(clear_raw.get("trim") or clear_raw.get("silenceTrim") or {}).get("enabled", True),
                    trim_silence_threshold=(clear_raw.get("trim") or clear_raw.get("silenceTrim") or {}).get("silenceThreshold", -40.0),
                    trim_min_silence_duration=(
                        (clear_raw.get("trim") or clear_raw.get("silenceTrim") or {}).get("minSilenceDuration")
                        or ((clear_raw.get("silenceTrim") or {}).get("minSilenceMs", 300) / 1000.0)
                    ),
                    chunking_enabled=(clear_raw.get("chunking") or {}).get("enabled", True),
                    chunking_max_duration_sec=(clear_raw.get("chunking") or {}).get("maxDurationSec")
                    or (clear_raw.get("chunking") or {}).get("maxDurationPerChunk", 30.0),
                    chunking_overlap_sec=(clear_raw.get("chunking") or {}).get("overlapSec")
                    or (clear_raw.get("chunking") or {}).get("overlap", 2.0),
                    metadata_include_fields=(clear_raw.get("metadata") or {}).get("includeFields", ["duration", "size", "snr_db"]),
                )
        else:
            # No clearAudio section configured → skip heavy audio processing.
            clear = None

        # Get CSV path from new structure or legacy
        csv_path = data.get("csv_path")
        if not csv_path and dev_config:
            # Try to get from dev config fallback
            dev_ingestion = dev_config.get("ingestion", {})
            csv_path = dev_ingestion.get("csv", {}).get("localFallbackPath")
        
        # Get S3 prefix from dev config if not in preprocessing
        s3_prefix = data.get("s3_prefix", "")
        if not s3_prefix and dev_config:
            dev_ingestion = dev_config.get("ingestion", {})
            s3_config = dev_ingestion.get("s3", {})
            s3_prefix = s3_config.get("rawPrefix", "").replace("/raw", "").rstrip("/")

        if not workflow_enabled:
            validation.enabled = False
            clear = None

        return PreprocessingConfig(
            csv_path=csv_path or "data/user_responses.csv",  # Final fallback
            s3_prefix=s3_prefix,
            add_timestamp=data.get("add_timestamp", False),
            version=data.get("version"),
            validation=validation,
            clear_audio=clear,
        )

    @staticmethod
    def _build_annotation(raw: Dict[str, Any], workflow_enabled: bool = True) -> AnnotationModuleConfig:
        if not raw:
            raise ValueError("annotation config is required")
        data = raw.copy()
        label_config = data.get("labelConfig") or data.get("label_config")
        if not label_config and not data.get("label_config_path"):
            raise ValueError("annotation.labelConfig or label_config_path is required")
        if not label_config:
            cfg_path = Path(data["label_config_path"])
            label_config = cfg_path.read_text(encoding="utf-8")

        pre_raw = data.get("preAnnotation")
        if pre_raw:
            # Optional explicit flag: preAnnotation.enabled: false → force-disable preannotation
            if pre_raw.get("enabled") is False:
                preannotation = None
            else:
                preannotation = PreAnnotationConfig(
                    enabled=pre_raw.get("enabled", True),
                    label_schema=LabelSchemaConfig(fields=(pre_raw.get("labelSchema") or {}).get("fields", ["transcription"])),
                )
        else:
            preannotation = None

        transcription_raw = data.get("transcription")
        if transcription_raw:
            # Optional explicit flag: transcription.enabled: false → skip ASR/transcription
            if transcription_raw.get("enabled") is False:
                transcription = None
            else:
                timestamps_raw = transcription_raw.get("timestamps") or transcription_raw.get("timestamp") or {}
                transcription = TranscriptionConfig(
                    asr=ASRConfig(**(transcription_raw.get("asr") or {})),
                    timestamp=TimestampConfig(format=timestamps_raw.get("format", "s")),
                    output=OutputConfig(**(transcription_raw.get("output") or {})),
                    sentiment_enabled=(transcription_raw.get("sentiment") or {}).get("enabled", False),
                )
        else:
            transcription = None

        enrichment_raw = data.get("additionalAnnotation") or data.get("enrichment")
        if enrichment_raw:
            # Optional explicit flag: additionalAnnotation.enabled: false → skip enrichment
            if enrichment_raw.get("enabled") is False:
                enrichment = None
            else:
                enrichment = AdditionalAnnotationConfig(
                    keywords=KeywordsConfig(**(enrichment_raw.get("keywords") or {})),
                    tagging=TaggingConfig(**(enrichment_raw.get("tagging") or {})),
                )
        else:
            enrichment = None

        ml_validation_raw = data.get("mlValidation") or data.get("ml_validation")
        if ml_validation_raw:
            ml_validation = MLValidationConfig(**ml_validation_raw)
        else:
            ml_validation = None

        template_raw = data.get("template") or {}
        mapping_rules = [
            TemplateMappingRule(source=rule.get("from"), target=rule.get("to"))
            for rule in (template_raw.get("mapping") or {}).get("rules", [])
            if rule.get("from") and rule.get("to")
        ]
        template = AnnotationTemplateConfig(
            template_id=template_raw.get("id", "default-template"),
            version=template_raw.get("version", "1.0"),
            mapping_rules=mapping_rules,
        )

        if not workflow_enabled:
            preannotation = None
            transcription = None
            enrichment = None
            ml_validation = None

        return AnnotationModuleConfig(
            label_config=label_config,
            preannotation=preannotation,
            transcription=transcription,
            enrichment=enrichment,
            ml_validation=ml_validation,
            template=template,
        )

    @staticmethod
    def _build_postprocessing(
        raw: Optional[Dict[str, Any]],
        dev_config: Optional[Dict[str, Any]] = None,
        workflow_enabled: bool = True,
    ) -> Optional[PostProcessingConfig]:
        if not raw:
            # Try to get from dev config
            if dev_config:
                dev_pp = dev_config.get("postProcessing", {})
                if dev_pp:
                    raw = dev_pp
                else:
                    return None
            else:
                return None
        
        data = raw.copy()

        # Optional explicit flag: postProcessing.enabled: false → disable entire postprocessing block
        if data.get("enabled") is False or not workflow_enabled:
            return None

        # Support nested "post" section, matching:
        #   post.merge.strategy
        #   post.schema.enabled / id
        #   post.output.formats[]
        #   post.s3.bucket / prefix
        #   post.completionWebhook.url / secret
        post_raw = data.pop("post", {}) or {}
        merge_raw = post_raw.get("merge") or {}
        schema_raw = post_raw.get("schema") or {}
        output_raw = post_raw.get("output") or {}
        post_s3_raw = post_raw.get("s3") or {}
        completion_raw = post_raw.get("completionWebhook") or data.get("completionWebhook") or {}
        
        # Also check dev config for completion webhook
        if dev_config:
            dev_pp = dev_config.get("postProcessing", {})
            if dev_pp and not completion_raw.get("url"):
                completion_raw = dev_pp.get("completionWebhook", completion_raw)
            if dev_pp and not post_s3_raw.get("prefix"):
                post_s3_raw = dev_pp.get("s3", post_s3_raw) or post_s3_raw

        # Backwards-compatible base fields
        base_kwargs: Dict[str, Any] = {
            "final_csv_path": data.get("final_csv_path", "./.tmp/final_report.csv"),
            "s3_prefix": data.get("s3_prefix"),
            "extract_audio_segments": data.get("extract_audio_segments", True),
        }

        # Get output_fields configuration (which fields to include in final JSON)
        output_fields = data.get("output_fields") or data.get("outputFields")
        if output_fields is None:
            # Default fields if not specified
            output_fields = ["start", "end", "speaker", "text", "language", "end_of_speech"]
        
        # New post-processing + completion webhook knobs
        post_kwargs: Dict[str, Any] = {
            "post_merge_strategy": merge_raw.get("strategy") or data.get("mergeStrategy", "prefer-human"),
            "post_schema_enabled": schema_raw.get("enabled", False),
            "post_schema_id": schema_raw.get("id"),
            "post_output_formats": output_raw.get("formats") or data.get("outputFormats", ["json", "csv"]),
            "post_s3_bucket": post_s3_raw.get("bucket"),
            "post_s3_prefix": post_s3_raw.get("prefix"),
            "completion_webhook_url": completion_raw.get("url"),
            "completion_webhook_secret": completion_raw.get("secret"),
            "output_fields": output_fields,
        }

        return PostProcessingConfig(**base_kwargs, **post_kwargs)

    @staticmethod
    def _build_rejection(data: Dict[str, Any]) -> Optional[RejectionHandlerConfig]:
        # Support both old structure and new devConfig structure
        raw = data.get("rejectionHandler")
        if not raw:
            # Also check webhooks section in devConfig
            webhooks_raw = data.get("webhooks", {})
            if webhooks_raw:
                # Build from webhooks section
                rejection_webhook = webhooks_raw.get("rejection") or webhooks_raw.get("labelStudioRejection", {})
                return RejectionHandlerConfig(
                    webhook=RejectionWebhookConfig(
                        url=rejection_webhook.get("url"),
                        secret=rejection_webhook.get("secret"),
                    ),
                    lambda_config=RejectionLambdaConfig(),
                    logging=RejectionLoggingConfig(),
                )
            return None
        logging_fallback = data.get("logging") or {}

        def _normalize(fields: List[str]) -> List[str]:
            # Legacy field name normalization (no longer needed, but kept for backward compatibility)
            return fields

        webhook = RejectionWebhookConfig(**(raw.get("webhook") or {}))

        lambda_raw = raw.get("lambda") or {}
        lambda_normalized = {
            "function_name": lambda_raw.get("function_name") or lambda_raw.get("functionName"),
            "payload_fields": lambda_raw.get("payload_fields")
            or lambda_raw.get("payloadFields")
            or [],
        }
        lambda_cfg = RejectionLambdaConfig(**lambda_normalized)

        logging_cfg = RejectionLoggingConfig(**(raw.get("logging") or logging_fallback))
        lambda_cfg.payload_fields = _normalize(lambda_cfg.payload_fields)
        logging_cfg.fields = _normalize(logging_cfg.fields)

        return RejectionHandlerConfig(
            webhook=webhook,
            lambda_config=lambda_cfg,
            logging=logging_cfg,
        )

    @staticmethod
    def _build_ls_project(data: Dict[str, Any]) -> LabelStudioProjectConfig:
        # Support both old structure and new devConfig structure
        project_raw = data.get("labelStudio") or data.get("lsProject")
        if not project_raw:
            raise ValueError("labelStudio/lsProject configuration is required (missing projectNamePrefix).")

        ml_backend_raw = project_raw.get("mlBackend") or project_raw.get("ml_backend") or {}
        export_raw = project_raw.get("export") or {}
        storage_raw = project_raw.get("storage") or {}
        webhook_raw = project_raw.get("webhook") or {}
        prefix = project_raw.get("projectNamePrefix") or project_raw.get("project_name_prefix")
        if not prefix:
            raise ValueError("lsProject.projectNamePrefix must be provided in modules.yaml")

        storage_config = StorageConfig(
            enabled=storage_raw.get("enabled", False),
            storage_type=storage_raw.get("type", storage_raw.get("storage_type", "s3")),
            bucket=storage_raw.get("bucket"),
            prefix=storage_raw.get("prefix"),
            region=storage_raw.get("region"),
            source_enabled=storage_raw.get("sourceEnabled", storage_raw.get("source_enabled", True)),
            source_prefix=storage_raw.get("sourcePrefix", storage_raw.get("source_prefix")),
            target_enabled=storage_raw.get("targetEnabled", storage_raw.get("target_enabled", False)),
            target_prefix=storage_raw.get("targetPrefix", storage_raw.get("target_prefix")),
            aws_access_key_id=storage_raw.get("awsAccessKeyId") or storage_raw.get("aws_access_key_id"),
            aws_secret_access_key=storage_raw.get("awsSecretAccessKey") or storage_raw.get("aws_secret_access_key"),
            use_blob_urls=storage_raw.get("useBlobUrls", storage_raw.get("use_blob_urls", True)),
            title=storage_raw.get("title"),
        )

        webhook_config = WebhookConfig(
            enabled=webhook_raw.get("enabled", False),
            url=webhook_raw.get("url"),
            events=webhook_raw.get("events", [
                "TASK_CREATED",
                "ANNOTATION_CREATED",
                "ANNOTATION_UPDATED",
                "TASK_COMPLETED",
            ]),
            is_active=webhook_raw.get("isActive", webhook_raw.get("is_active", True)),
        )

        return LabelStudioProjectConfig(
            project_name_prefix=prefix,
            description=project_raw.get("description", ""),
            template_id=project_raw.get("templateId", project_raw.get("template_id")),
            batch_size=project_raw.get("batchSize", project_raw.get("batch_size", 50)),
            ml_backend=MLBackendConfig(
                enabled=ml_backend_raw.get("enabled", True),
                url=ml_backend_raw.get("url"),
                token=ml_backend_raw.get("token"),
            ),
            storage=storage_config,
            webhook=webhook_config,
            run_pre_annotate=project_raw.get("runPreAnnotate", project_raw.get("run_pre_annotate", True)),
            reviewer_ids=project_raw.get("reviewerIds", project_raw.get("reviewer_ids", [])),
            export=LSExportConfig(**export_raw or {}),
        )
