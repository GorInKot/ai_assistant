"""add timing_runs

Замер времени задач «вручную / с ассистентом» (страница «Замер»).
Только добавление таблицы, существующие данные не затрагиваются.

Revision ID: b7d2e4f91a63
Revises: f3c8a6d21b47
Create Date: 2026-10-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7d2e4f91a63'
down_revision: Union[str, Sequence[str], None] = 'f3c8a6d21b47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'timing_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('participant', sa.String(), nullable=False),
        sa.Column('task', sa.String(), nullable=False),
        sa.Column('document', sa.String(), nullable=False),
        sa.Column('doc_type', sa.String(), nullable=False),
        sa.Column('mode', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('machine_seconds', sa.Float(), server_default='0', nullable=False),
        sa.Column('sheets', sa.Integer(), nullable=True),
        sa.Column('positions', sa.Integer(), nullable=True),
        sa.Column('flagged', sa.Integer(), nullable=True),
        sa.Column('errors_wrong', sa.Integer(), nullable=True),
        sa.Column('errors_missed', sa.Integer(), nullable=True),
        sa.Column('errors_extra', sa.Integer(), nullable=True),
        sa.Column('changes_missed', sa.Integer(), nullable=True),
        sa.Column('changes_false', sa.Integer(), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_timing_runs_id'), 'timing_runs', ['id'], unique=False)
    op.create_index(op.f('ix_timing_runs_created_by'), 'timing_runs', ['created_by'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_timing_runs_created_by'), table_name='timing_runs')
    op.drop_index(op.f('ix_timing_runs_id'), table_name='timing_runs')
    op.drop_table('timing_runs')
