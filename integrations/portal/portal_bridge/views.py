from accounts.access import user_is_admin, user_is_manager_or_admin
from accounts.models import Profile
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http.request import split_domain_port
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.cache import never_cache
from django.views.decorators.clickjacking import xframe_options_sameorigin

from portalrnsk import views as portal_views

from .sso_pass import make_pass, subject

MIN_SECRET_LEN = 32
GRID_MARKER = '<section class="grid" aria-label="Разделы портала">'
TILE = """
      <a class="panel {accent}" href="{url}">
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        <span class="status">Доступно</span>
      </a>"""
# Плитки надстройки: (адрес портала, цвет, заголовок, описание).
TILES = (
    ("assistant", "yellow", "ИИ-ассистент",
     "Ответы по регламентам с источниками, распознавание и сравнение документов."),
    ("assistant_tools", "cyan", "Конструктор и чертежи",
     "3D-модель простой детали и эскиз трассы трубопровода по описанию словами."),
)


@login_required
def home(request):
    """Главная портала как есть, плюс плитки ассистента первыми в сетке разделов."""
    response = portal_views.home(request)
    html = response.content.decode(response.charset)
    if GRID_MARKER in html:  # разметку главной поменяли — оставляем страницу как есть
        tiles = "".join(
            TILE.format(url=reverse(name), accent=accent, title=title, description=description)
            for name, accent, title, description in TILES
        )
        response.content = html.replace(GRID_MARKER, GRID_MARKER + tiles, 1).encode(response.charset)
    return response


def assistant_base_url(scheme: str, host: str) -> str:
    """Адрес ассистента; без ASSISTANT_URL — тот же хост, что у портала, порт ASSISTANT_PORT."""
    if settings.ASSISTANT_URL:
        return settings.ASSISTANT_URL
    domain, _port = split_domain_port(host)
    return f"{scheme}://{domain}:{settings.ASSISTANT_PORT}"


def _role(user) -> str:
    if user_is_admin(user):
        return "admin"
    if user_is_manager_or_admin(user):
        return "manager"
    return "user"


def _sso_context(request, *, next_path: str) -> dict:
    """Одноразовый пропуск в ассистент; next_path — раздел ассистента после входа
    (ассистент принимает только разделы из своего списка, см. portal_sso.ALLOWED_NEXT)."""
    profile, _created = Profile.objects.get_or_create(user=request.user)
    context = {
        "profile": profile,
        "assistant_url": assistant_base_url(request.scheme, request.get_host()),
        "token": "",
        "next_path": next_path,
    }
    secret = settings.ASSISTANT_SSO_SECRET
    if len(secret) >= MIN_SECRET_LEN:
        user = request.user
        context["token"] = make_pass(
            sub=subject(user.username, user.pk),
            last_name=user.last_name or "",
            first_name=user.first_name or "",
            middle_name=profile.middle_name or "",
            role=_role(user),
            secret=secret,
        )
    return context


@never_cache
@login_required
def assistant(request):
    context = _sso_context(request, next_path="/")
    # «Открыть полностью» из панели: та же беседа в полной версии.
    conversation = request.GET.get("conversation", "")
    context["conversation"] = conversation if conversation.isdigit() and len(conversation) <= 12 else ""
    return render(request, "assistant/index.html", {**context, "title": "ИИ-ассистент"})


@never_cache
@login_required
def assistant_tools(request):
    context = _sso_context(request, next_path="/tools")
    return render(request, "assistant/index.html", {**context, "title": "Конструктор и чертежи"})


@never_cache
@xframe_options_sameorigin
@login_required
def assistant_panel(request):
    """Содержимое выезжающей панели (widget.js): пропуск сразу уходит в ассистент, на /widget.
    host — адрес портала, на который панель шлёт сообщения («закрыть», «открыть полностью»)."""
    context = _sso_context(request, next_path="/widget")
    context["host"] = f"{request.scheme}://{request.get_host()}"
    return render(request, "assistant/panel.html", context)
