from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SEMANTIC_REVIEW_ATTACHMENT = "semantic_review.json"
ATOMIC_TOPIC = "/semantic/atomic_state"
SUBTASK_TOPIC = "/semantic/subtask_state"
JSON_SENSOR_TOPICS = (
    "/tactile/raw/left",
    "/tactile/raw/right",
    "/hand_skeleton/left",
    "/hand_skeleton/right",
    "/pose/head",
    "/pose/wrist/left",
    "/pose/wrist/right",
    "/time_sync/info",
)
VIDEO_TOPICS = ("/ego_video/head", "/ego_video/left", "/ego_video/right")
VIEW_BY_TOPIC = {topic: index for index, topic in enumerate(VIDEO_TOPICS)}


class McapDependencyError(RuntimeError):
    pass


def _make_reader(handle: Any, *, protobuf: bool = False) -> Any:
    try:
        from mcap.reader import make_reader
    except ImportError as error:  # pragma: no cover - exercised without optional extra
        raise McapDependencyError(
            "MCAP support is required. Install the project with the 'mcap' extra."
        ) from error
    if not protobuf:
        return make_reader(handle)
    try:
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as error:  # pragma: no cover
        raise McapDependencyError(
            "Video extraction requires mcap-protobuf-support."
        ) from error
    return make_reader(handle, decoder_factories=[DecoderFactory()])


@dataclass(frozen=True)
class McapSummary:
    schemas: tuple[str, ...]
    topics: tuple[str, ...]
    topic_message_counts: Mapping[str, int]
    attachment_names: tuple[str, ...]
    message_start_time_ns: int
    message_end_time_ns: int
    message_count: int


def read_summary(path: str | Path) -> McapSummary:
    source = Path(path)
    with source.open("rb") as handle:
        reader = _make_reader(handle)
        summary = reader.get_summary()
    if summary is None or summary.statistics is None:
        raise ValueError(f"MCAP lacks a readable summary: {source}")
    topic_counts = {
        summary.channels[channel_id].topic: int(count)
        for channel_id, count in summary.statistics.channel_message_counts.items()
    }
    return McapSummary(
        schemas=tuple(sorted(schema.name for schema in summary.schemas.values())),
        topics=tuple(sorted(channel.topic for channel in summary.channels.values())),
        topic_message_counts=dict(sorted(topic_counts.items())),
        attachment_names=tuple(sorted(index.name for index in summary.attachment_indexes)),
        message_start_time_ns=int(summary.statistics.message_start_time),
        message_end_time_ns=int(summary.statistics.message_end_time),
        message_count=int(summary.statistics.message_count),
    )


def read_attachment_json(
    path: str | Path, name: str = SEMANTIC_REVIEW_ATTACHMENT
) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        reader = _make_reader(handle)
        matches = [attachment for attachment in reader.iter_attachments() if attachment.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name!r} attachment in {source}, got {len(matches)}")
    try:
        value = json.loads(matches[0].data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Attachment {name!r} is not valid UTF-8 JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Attachment {name!r} must contain a JSON object: {source}")
    return value


def iter_json_messages(
    path: str | Path,
    topics: Sequence[str] | None = None,
) -> Iterator[tuple[str, int, dict[str, Any], str | None]]:
    source = Path(path)
    with source.open("rb") as handle:
        reader = _make_reader(handle)
        for schema, channel, message in reader.iter_messages(topics=topics):
            if channel.message_encoding != "json":
                raise ValueError(
                    f"Topic {channel.topic} uses {channel.message_encoding}, expected json"
                )
            try:
                value = json.loads(message.data)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid JSON on {channel.topic} at {message.log_time}: {source}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"JSON message on {channel.topic} is not an object")
            yield (
                channel.topic,
                int(message.log_time),
                value,
                schema.name if schema is not None else None,
            )


def read_first_json_by_topic(
    path: str | Path, topics: Sequence[str]
) -> dict[str, dict[str, Any]]:
    wanted = set(topics)
    output: dict[str, dict[str, Any]] = {}
    for topic, _, value, _ in iter_json_messages(path, topics=topics):
        output.setdefault(topic, value)
        if wanted.issubset(output):
            break
    return output


def canonical_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    name = segment.get("name", segment.get("label"))
    if not isinstance(name, str) or not name:
        raise ValueError(f"Atomic segment has no non-empty name: {segment}")
    start = segment.get("start_frame_60")
    end = segment.get("end_frame_60_exclusive")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise ValueError(f"Invalid half-open atomic interval: {segment}")
    return {
        "name": name,
        "hand": str(segment.get("hand", "")),
        "start_frame_60": int(start),
        "end_frame_60_exclusive": int(end),
    }


def read_dual_source_labels(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    attachment = read_attachment_json(path)
    attachment_segments = [
        canonical_segment(value) for value in attachment.get("atomic_segments", [])
    ]
    topic_segments = [
        canonical_segment(value)
        for topic, _, value, _ in iter_json_messages(path, topics=[ATOMIC_TOPIC])
        if topic == ATOMIC_TOPIC
    ]
    return attachment, attachment_segments, topic_segments


def assert_dual_source_labels_equal(
    attachment_segments: Sequence[Mapping[str, Any]],
    topic_segments: Sequence[Mapping[str, Any]],
    *,
    source: str | Path | None = None,
) -> None:
    if len(attachment_segments) != len(topic_segments):
        raise ValueError(
            f"Atomic label count mismatch for {source}: attachment={len(attachment_segments)}, "
            f"topic={len(topic_segments)}"
        )
    for index, (attachment, topic) in enumerate(
        zip(attachment_segments, topic_segments, strict=True)
    ):
        left = canonical_segment(attachment)
        right = canonical_segment(topic)
        if left != right:
            raise ValueError(
                f"Atomic label mismatch for {source} at segment {index}: "
                f"attachment={left}, topic={right}"
            )


def iter_decoded_images(
    path: str | Path,
    topics: Sequence[str] = VIDEO_TOPICS,
) -> Iterator[tuple[str, int, bytes, str, Any]]:
    """Yield decoded protobuf image payloads without materializing the MCAP.

    The MCAP log time is the session clock used for cross-modal alignment.  The
    protobuf acquisition timestamp is returned only as provenance.
    """

    source = Path(path)
    with source.open("rb") as handle:
        reader = _make_reader(handle, protobuf=True)
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
            image_format = str(getattr(decoded, "format", ""))
            data = bytes(getattr(decoded, "data"))
            acquisition_timestamp = getattr(decoded, "timestamp", None)
            yield channel.topic, int(message.log_time), data, image_format, acquisition_timestamp


def iter_episode_json_streams(
    path: str | Path,
    topics: Iterable[str] = JSON_SENSOR_TOPICS,
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    selected = tuple(topics)
    for topic, log_time, value, _ in iter_json_messages(path, topics=selected):
        yield topic, log_time, value
