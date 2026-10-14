"""add kind to cad_requests

Тип запроса в истории конструктора: part — деталь (3D), route — чертёж трассы (2D).
Существующие записи — детали. Только добавление колонки.

Revision ID: f3c8a6d21b47
Revises: e8a1c4d97f26
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3c8a6d21b47'
down_revision: Union[str, Sequence[str], None] = 'e8a1c4d97f26'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('cad_requests', sa.Column('kind', sa.String(), nullable=False, server_default='part'))


def downgrade() -> None:
    with op.batch_alter_table('cad_requests', schema=None) as batch_op:
        batch_op.drop_column('kind')
