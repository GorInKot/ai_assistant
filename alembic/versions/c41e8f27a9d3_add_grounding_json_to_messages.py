"""add grounding_json to messages

Проверка обоснованности ответа (цитаты и неподтверждённые утверждения) хранится JSON-ом
в сообщении ассистента, чтобы при повторном открытии беседы цитаты не пропадали.
Колонка необязательная: старые сообщения остаются как есть. Данные не затрагиваются.

Revision ID: c41e8f27a9d3
Revises: a7c3d9e15b42
Create Date: 2026-10-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c41e8f27a9d3'
down_revision: Union[str, Sequence[str], None] = 'a7c3d9e15b42'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('messages', sa.Column('grounding_json', sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_column('grounding_json')
