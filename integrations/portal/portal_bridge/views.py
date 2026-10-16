from accounts.access import user_is_admin, user_is_manager_or_admin
from accounts.models import Profile
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http.request import split_domain_port
from django.shortcuts import render
from django.urls import reverse

from portalrnsk import views as portal_views

from .sso_pass import make_pass, subject

MIN_SECRET_LEN = 32
GRID_MARKER = '<section class="grid" aria-label="Разделы портала">'
TILE = """
      <a class="panel yellow" href="{url}">
        <div>
          <h2>ИИ-ассистент</h2>
          <p>Ответы по регламентам с источниками, распознавание и сравнение документов.</p>
        </div>
        <span class="status">Доступно</span>
      </a>"""


@login_required
def home(request):
    """Главная портала как есть, плюс плитка ассистента первой в сетке разделов."""
    response = portal_views.home(request)
    html = response.content.decode(response.charset)
    if GRID_MARKER in html:  # разметку главной поменяли — оставляем страницу как есть
        tile = TILE.format(url=reverse("assistant"))
        response.content = html.replace(GRID_MARKER, GRID_MARKER + tile, 1).encode(response.charset)
    return response


def _assistant_url(request) -> str:
    if settings.ASSISTANT_URL:
        return settings.ASSISTANT_URL
    domain, _port = split_domain_port(request.get_host())
    return f"{request.scheme}://{domain}:{settings.ASSISTANT_PORT}"


def _role(user) -> str:
    if user_is_admin(user):
        return "admin"
    if user_is_manager_or_admin(user):
        return "manager"
    return "user"


@login_required
def assistant(request):
    profile, _created = Profile.objects.get_or_create(user=request.user)
    context = {"profile": profile, "assistant_url": _assistant_url(request), "token": ""}
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
    return render(request, "assistant/index.html", context)
