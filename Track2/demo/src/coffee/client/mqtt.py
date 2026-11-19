"""Publish one MQTT message per image. The paho client is created only when connecting."""

from __future__ import annotations

import logging
from collections.abc import Callable

from coffee.config import AppConfig, ConfigError

logger = logging.getLogger("coffee.client.mqtt")


class Publisher:
    def __init__(self, cfg: AppConfig, connect: Callable[[], object] | None = None) -> None:
        if not cfg.mqtt_topic:
            raise ConfigError("MQTT topic is empty. Set COFFEE_MQTT_TOPIC.")
        self._cfg = cfg
        self._connect = connect or self._paho
        self._client: object | None = None

    def __enter__(self) -> Publisher:
        self._client = self._connect()
        return self

    def __exit__(self, *exc: object) -> None:
        client = self._client
        self._client = None
        if client is not None and hasattr(client, "loop_stop"):
            client.loop_stop()
            client.disconnect()

    def publish(self, payload: bytes) -> None:
        if self._client is None:
            raise ConfigError("MQTT client is not connected.")
        topic = resolve_topic(self._cfg)
        info = self._client.publish(topic, payload, qos=1)
        info.wait_for_publish(timeout=self._cfg.mqtt_timeout_s)
        if not info.is_published():
            raise ConfigError(f"MQTT publish to {topic} was not acknowledged.")

    def _paho(self) -> object:
        import paho.mqtt.client as mqtt

        client_id = self._cfg.mqtt_client_id or self._cfg.team
        secure = self._cfg.mqtt_port == 443
        if secure:
            import ssl

            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=client_id,
                transport="websockets",
                protocol=mqtt.MQTTv5,
            )
            client.ws_set_options(path=self._cfg.mqtt_path or "/mqtt/")
            client.tls_set_context(ssl.create_default_context())
        else:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        if self._cfg.mqtt_username or self._cfg.mqtt_password:
            client.username_pw_set(self._cfg.mqtt_username, self._cfg.mqtt_password)
        client.connect(self._cfg.mqtt_host, self._cfg.mqtt_port, keepalive=30)
        client.loop_start()
        logger.info(
            "MQTT connected to %s:%s topic %s client %s",
            self._cfg.mqtt_host,
            self._cfg.mqtt_port,
            resolve_topic(self._cfg),
            client_id,
        )
        return client


def resolve_topic(cfg: AppConfig) -> str:
    """Fill leftover TRACK/EVENT placeholders. A finished topic is returned unchanged."""
    topic = cfg.mqtt_topic.strip()
    topic = topic.replace("/TRACK/", f"/{cfg.mqtt_track}/")
    if topic.endswith("/EVENT"):
        topic = f"{topic[: -len('EVENT')]}{cfg.mqtt_event}"
    return topic
