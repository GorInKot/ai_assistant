"""add cad_requests

История запросов во вкладке «Конструктор деталей»: описание, исход и спецификация детали.
Только добавление таблицы, существующие данные не затрагиваются.

Revision ID: e8a1c4d97f26
Revises: d52f6b83c1e7
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8a1c4d97f26'
down_revision: Union[str, Sequence[str], None] = 'd52f6b83c1e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'cad_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('outcome', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('detail', sa.Text(), nullable=True),
        sa.Column('spec_json', sa.Text(), nullable=True),
        sa.Column('notes_json', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_cad_requests_id'), 'cad_requests', ['id'], unique=False)
    op.create_index(op.f('ix_cad_requests_user_id'), 'cad_requests', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_cad_requests_user_id'), table_name='cad_requests')
    op.drop_index(op.f('ix_cad_requests_id'), table_name='cad_requests')
    op.drop_table('cad_requests')
