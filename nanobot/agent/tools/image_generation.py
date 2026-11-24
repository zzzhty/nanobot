from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.image_generation_config import ImageGenerationToolConfig
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.schema import (
    ArraySchema,
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.config.network import NetworkConfig
from nanobot.config.paths import get_media_dir
from nanobot.providers.image_generation import (
    ImageGenerationError,
    ImageGenerationProvider,
    get_image_gen_provider,
    image_gen_provider_configs,
)
from nanobot.security.workspace_access import current_tool_workspace
from nanobot.security.workspace_policy import WorkspaceBoundaryError, resolve_allowed_path
from nanobot.utils.artifacts import (
    ArtifactError,
    generated_image_tool_result,
    store_generated_image_artifact,
)
from nanobot.utils.helpers import detect_image_mime

_SECRET_RE = re.compile(r"(?i)\b(?:api[_-]?key|token|secret|bearer)\b\s*[:=]\s*\S+")
_URL_RE = re.compile(r"https?://\S+")


class ImageRequestRejectedError(Exception):
    """The provider finished and refused this request or returned no usable image."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ImageRequestUnresolvedError(Exception):
    """The call ended without proof that the provider accepted or rejected it."""


def public_image_reason(exc: BaseException) -> str:
    text = " ".join(_URL_RE.sub("", _SECRET_RE.sub("", str(exc))).split())
    if not text:
        return "生图服务返回了失败结果"
    return text[:160]


def _transport_gave_up(exc: BaseException) -> bool:
    """True when the call ended without a provider response.

    A status code means the service answered. A timeout or dropped connection
    does not prove that the service never started the job.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            return False
        if isinstance(current, (httpx.TimeoutException, httpx.RequestError, TimeoutError, ConnectionError)):
            return True
        current = current.__cause__
    return False


if TYPE_CHECKING:
    from nanobot.agent.tools.context import ToolContext
    from nanobot.config.schema import Config, ProviderConfig


_IMAGE_PARAMETERS = tool_parameters_schema(
    prompt=StringSchema(
        "Detailed image generation or edit instructions.",
        min_length=1,
    ),
    reference_images=ArraySchema(
        StringSchema(""),
        description="Existing local image paths for edits, including prior generated artifacts.",
    ),
    aspect_ratio=StringSchema(
        "Output aspect ratio, e.g. 1:1 or 16:9.",
    ),
    image_size=StringSchema(
        "Provider-supported output size hint, e.g. 1K or 1024x1024.",
    ),
    count=IntegerSchema(
        description="Number of images to generate in this turn.",
        minimum=1,
        maximum=8,
    ),
    required=["prompt"],
)


def _queue_parameters() -> dict[str, Any]:
    schema = deepcopy(_IMAGE_PARAMETERS)
    properties = schema["properties"]
    properties["queue_mode"] = StringSchema(
        "For the gateway queue: status to inspect drafts; merge (default) with a full updated brief; "
        "new for a separate image; recover to read interrupted inputs; defer to hold a draft for clarification; "
        "wait to keep observing the same running provider call. "
        "wait does not generate again and does not start a billed retry. "
        "status.can_wait is false when that call has finished, was cancelled, or the process restarted. "
        "An unconfirmed provider call is not retried and does not block a later draft.",
        enum=["status", "merge", "new", "recover", "defer", "wait"],
    ).to_json_schema()
    properties["draft_id"] = StringSchema(
        "ID from status. A draft is edited; a frozen image may seed the next draft.",
    ).to_json_schema()
    schema["required"] = []
    return schema


class ImageGenerationTool(Tool):
    """Generate persistent image artifacts through the configured image provider."""

    config_key = "image_generation"

    @classmethod
    def config_cls(cls):
        return ImageGenerationToolConfig

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.providers_config is not None and ctx.config.image_generation.enabled

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        return cls(
            workspace=ctx.workspace,
            config=ctx.config.image_generation,
            provider_configs=image_gen_provider_configs(ctx.providers_config),
            network=ctx.network,
        )

    def __init__(
        self,
        *,
        workspace: str | Path,
        config: ImageGenerationToolConfig,
        provider_config: ProviderConfig | None = None,
        provider_configs: dict[str, ProviderConfig] | None = None,
        network: NetworkConfig | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser()
        self.config = config
        self.network = network or NetworkConfig()
        self.provider_configs = dict(provider_configs or {})
        if provider_config is not None and "openrouter" not in self.provider_configs:
            self.provider_configs["openrouter"] = provider_config

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        if self._queue_enabled():
            return (
                "Queue image generation for the current interactive QQ or WebUI turn. "
                "The gateway delivers those results later. On every other turn, generate the image immediately. "
                "Use status before modifying a pending image. Do not claim completion or submit twice. "
                "Use new only for a separate request. "
                "If the user asks to keep waiting for an image that is already running, use queue_mode=wait."
            )
        return (
            "Generate or edit images as persistent artifacts. Returns artifact IDs and local paths."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        if self._queue_enabled():
            return _queue_parameters()
        return deepcopy(_IMAGE_PARAMETERS)

    def _queue_enabled(self) -> bool:
        if os.environ.get("NANOBOT_IMAGE_QUEUE") == "1":
            return True
        return bool(self.config.persistent_queue)

    def _provider_config(self) -> ProviderConfig | None:
        return self.provider_configs.get(self.config.provider)

    def _provider_client(self) -> ImageGenerationProvider | None:
        provider = self._provider_config()
        cls = get_image_gen_provider(self.config.provider, provider)
        if cls is None:
            return None
        headers = dict(provider.extra_headers or {}) if provider else {}
        overridden = {key.lower() for key in (self.config.extra_headers or {})}
        headers = {key: value for key, value in headers.items() if key.lower() not in overridden}
        headers.update(self.config.extra_headers or {})
        body = deepcopy(provider.extra_body or {}) if provider else {}
        body.update(deepcopy(self.config.extra_body or {}))
        kwargs: dict[str, Any] = {
            "api_key": provider.api_key if provider and isinstance(provider.api_key, str) else None,
            "api_base": provider.api_base if provider and isinstance(provider.api_base, str) else None,
            "extra_headers": headers or None,
            "extra_body": body or None,
            "proxy": self.network.resolve_proxy(provider.use_proxy) if provider else None,
        }
        return cls(**kwargs)

    def _resolve_reference_image(self, value: str) -> str:
        access = current_tool_workspace(self.workspace, restrict_to_workspace=True)
        workspace = access.project_path or self.workspace
        try:
            resolved = resolve_allowed_path(
                value,
                workspace=workspace,
                allowed_root=access.allowed_root,
                extra_allowed_roots=[get_media_dir()] if access.allowed_root is not None else None,
                strict=True,
            )
        except WorkspaceBoundaryError as exc:
            raise ImageGenerationError(
                "reference_images must be inside the workspace or nanobot media directory"
            ) from exc
        except OSError as exc:
            raise ImageGenerationError(f"reference image not found: {value}") from exc
        if not resolved.is_file():
            raise ImageGenerationError(f"reference image is not a file: {value}")
        raw = resolved.read_bytes()
        if detect_image_mime(raw) is None:
            raise ImageGenerationError(f"unsupported reference image: {value}")
        return str(resolved)

    def resolve_reference_images(self, values: list[str] | None) -> list[str]:
        if not values:
            return []
        return [self._resolve_reference_image(value) for value in values if value]

    async def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str | None = None,
        reference_images: list[str] | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
        count: int | None = None,
        queue_mode: str = "merge",
        draft_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        if self._queue_enabled():
            from nanobot.agent.image_queue import handle_image_request, interactive_queue_turn

            if interactive_queue_turn():
                return await handle_image_request(
                    self,
                    prompt,
                    reference_images,
                    aspect_ratio,
                    image_size,
                    count,
                    queue_mode,
                    draft_id,
                )
            if queue_mode not in {"merge", "new"}:
                return ToolResult.error(
                    "Queued image controls require an interactive QQ or WebUI user turn. "
                    "Other turns generate the image directly."
                )
        if not prompt or queue_mode not in {"merge", "new"}:
            return ToolResult.error(
                "Error: a prompt is required; queue controls are only available when the persistent image queue is enabled"
            )
        return await self.generate_now(prompt, reference_images, aspect_ratio, image_size, count)

    async def generate_now(
        self,
        prompt: str,
        reference_images: list[str] | None,
        aspect_ratio: str | None,
        image_size: str | None,
        count: int | None,
        *,
        references_are_resolved: bool = False,
        propagate_provider_errors: bool = False,
    ) -> str:
        client = self._provider_client()
        if client is None:
            return ToolResult.error(f"Error: unsupported image generation provider '{self.config.provider}'")

        requested = count or 1
        if requested > self.config.max_images_per_turn:
            return ToolResult.error(
                "Error: count exceeds tools.imageGeneration.maxImagesPerTurn "
                f"({self.config.max_images_per_turn})"
            )

        try:
            refs = (
                list(reference_images or [])
                if references_are_resolved
                else self.resolve_reference_images(reference_images)
            )
            artifacts: list[dict[str, Any]] = []
            while len(artifacts) < requested:
                response = await client.generate(
                    prompt=prompt,
                    model=self.config.model,
                    reference_images=refs,
                    aspect_ratio=aspect_ratio or self.config.default_aspect_ratio,
                    image_size=image_size or self.config.default_image_size,
                )
                for image_data_url in response.images:
                    artifact = store_generated_image_artifact(
                        image_data_url,
                        prompt=prompt,
                        model=self.config.model,
                        source_images=refs,
                        save_dir=self.config.save_dir,
                        provider=self.config.provider,
                    )
                    artifacts.append(artifact)
                    if len(artifacts) >= requested:
                        break
            return generated_image_tool_result(artifacts)
        except ImageGenerationError as exc:
            if propagate_provider_errors:
                if _transport_gave_up(exc):
                    raise ImageRequestUnresolvedError() from exc
                raise ImageRequestRejectedError(public_image_reason(exc)) from exc
            return ToolResult.error(f"Error: {exc}")
        except ArtifactError as exc:
            if propagate_provider_errors:
                raise ImageRequestRejectedError("图片没有保存下来") from exc
            return ToolResult.error(f"Error: {exc}")
        except httpx.HTTPStatusError as exc:
            if propagate_provider_errors:
                raise ImageRequestRejectedError(public_image_reason(exc)) from exc
            return ToolResult.error(f"Error: {exc}")
        except (OSError, httpx.HTTPError) as exc:
            if propagate_provider_errors:
                raise ImageRequestUnresolvedError() from exc
            return ToolResult.error(f"Error: {exc}")


def register_image_generation_tool(
    config: Config,
    registry: ToolRegistry,
    *,
    workspace: Path,
) -> None:
    """Assemble the configured tool in the application's shared registry."""
    from nanobot.agent.tools.context import ToolContext

    ctx = ToolContext(
        config=config.tools,
        network=config.network,
        workspace=str(workspace),
        providers_config=config.providers,
    )
    if ImageGenerationTool.enabled(ctx):
        registry.register(ImageGenerationTool.create(ctx))
    else:
        registry.unregister("generate_image")


async def reload_image_generation_tool(
    *,
    config_path: Path,
    workspace: Path,
    registry: ToolRegistry,
) -> dict[str, Any]:
    """Refresh the application's image tool from its persisted config."""
    try:
        from nanobot.config.loader import load_config, resolve_config_env_vars

        config = resolve_config_env_vars(load_config(config_path), config_path=config_path)
        register_image_generation_tool(config, registry, workspace=workspace)
    except Exception as exc:
        logger.warning("Image generation hot reload could not read config: {}", exc)
        return {
            "ok": False,
            "message": "Could not reload image generation config.",
            "requires_restart": True,
            "error": str(exc),
        }

    tool_config = config.tools.image_generation
    logger.info(
        "Image generation config reloaded: enabled={} provider={} model={}",
        tool_config.enabled,
        tool_config.provider,
        tool_config.model,
    )
    return {
        "ok": True,
        "message": "Image generation settings applied without restarting nanobot.",
        "enabled": tool_config.enabled,
        "provider": tool_config.provider,
        "model": tool_config.model,
        "requires_restart": False,
    }
