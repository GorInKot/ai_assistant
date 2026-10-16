"""Закрытый контур: ассистент обращается только к этой машине и внутренней сети.

- model_endpoint_allowed() — проверка адресов моделей при старте (app/state.py). Адрес вне
  контура (облачный сервис, в том числе «адрес не задан, но есть ключ» — тогда клиент OpenAI
  идёт в облако по умолчанию) — модель отключается, пока это не разрешено явно
  ALLOW_EXTERNAL_MODELS=1. Сервер с моделью во внутренней сети: IP из частных диапазонов
  или имя из INTERNAL_HOSTS (через запятую).
- no_external_connections() — перехват сетевых соединений процесса: всё, кроме этой
  машины, блокируется и попадает в журнал. Для автотеста и scripts/check_offline.py.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse


def _internal_names() -> set[str]:
    names = {h.strip().lower() for h in os.getenv("INTERNAL_HOSTS", "").split(",") if h.strip()}
    return {"localhost"} | names


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def is_internal_host(host: str | None) -> bool:
    """Эта машина или внутренняя сеть: loopback, частные диапазоны IP, localhost, INTERNAL_HOSTS."""
    if not host:
        return False
    ip = _ip(host)
    if ip is None:
        return host.lower() in _internal_names()
    return ip.is_loopback or ip.is_private


def is_internal_url(url: str) -> bool:
    return is_internal_host(urlparse(url).hostname)


def external_models_allowed() -> bool:
    return os.getenv("ALLOW_EXTERNAL_MODELS", "").strip().lower() in {"1", "true", "yes", "on"}


def model_endpoint_allowed(base_url: str | None, api_key: str | None) -> tuple[bool, str]:
    """Можно ли обращаться к модели с такими настройками: (да/нет, пояснение для журнала)."""
    if not base_url and not api_key:
        return True, "модель не настроена"
    if external_models_allowed():
        return True, "внешние модели разрешены ALLOW_EXTERNAL_MODELS"
    if not base_url:
        return False, "адрес не задан — клиент обратился бы к облачному сервису по умолчанию"
    if is_internal_url(base_url):
        return True, "адрес внутри контура"
    return False, f"адрес {urlparse(base_url).hostname} вне контура (не эта машина и не внутренняя сеть)"


# ------------------------- перехват соединений -------------------------------


@dataclass
class NetLog:
    """Журнал перехвата: заблокированные попытки выйти за пределы машины и локальные соединения."""

    blocked: list[str] = field(default_factory=list)
    local: list[str] = field(default_factory=list)


def _is_loopback(host: str) -> bool:
    ip = _ip(host)
    return ip.is_loopback if ip is not None else host.lower() == "localhost"


@contextmanager
def no_external_connections():
    """Внутри блока процесс может соединяться только с этой машиной (loopback, сокеты Unix).
    Попытка соединения или разрешения имени вне машины — OSError и запись в журнал."""
    log = NetLog()
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_sendto, real_getaddrinfo = socket.socket.sendto, socket.getaddrinfo

    def check(address) -> None:
        if not isinstance(address, tuple) or not address:
            return  # сокет Unix — внутри машины
        host = str(address[0])
        label = f"{host}:{address[1]}" if len(address) > 1 else host
        if _is_loopback(host):
            log.local.append(label)
            return
        log.blocked.append(label)
        raise OSError(f"Соединение вне машины запрещено: {label}")

    def connect(sock, address):
        check(address)
        return real_connect(sock, address)

    def connect_ex(sock, address):
        check(address)
        return real_connect_ex(sock, address)

    def sendto(sock, data, *args):
        check(args[-1])
        return real_sendto(sock, data, *args)

    def getaddrinfo(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else host
        # IP-адрес разрешается без DNS — его остановит проверка соединения
        if name and _ip(str(name)) is None and not _is_loopback(str(name)):
            log.blocked.append(f"DNS {name}")
            raise socket.gaierror(f"Разрешение имён вне машины запрещено: {name}")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
    socket.socket.sendto, socket.getaddrinfo = sendto, getaddrinfo
    try:
        yield log
    finally:
        socket.socket.connect, socket.socket.connect_ex = real_connect, real_connect_ex
        socket.socket.sendto, socket.getaddrinfo = real_sendto, real_getaddrinfo
