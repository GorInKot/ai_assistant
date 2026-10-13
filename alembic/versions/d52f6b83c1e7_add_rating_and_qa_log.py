"""add message rating and qa_log

Оценка ответа («полезно / не полезно») и журнал вопросов к базе знаний для аналитики
пробелов. Только добавление: существующие данные не затрагиваются.

Revision ID: d52f6b83c1e7
Revises: c41e8f27a9d3
Create Date: 2026-10-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd52f6b83c1e7'
down_revision: Union[str, Sequence[str], None] = 'c41e8f27a9d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('messages', sa.Column('rating', sa.Integer(), nullable=True))
    op.create_table(
        'qa_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('message_id', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('division', sa.String(), nullable=True),
        sa.Column('claims_total', sa.Integer(), nullable=True),
        sa.Column('claims_unsupported', sa.Integer(), nullable=True),
        sa.Column('sources_json', sa.Text(), nullable=True),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_qa_log_id'), 'qa_log', ['id'], unique=False)
    op.create_index(op.f('ix_qa_log_created_at'), 'qa_log', ['created_at'], unique=False)
    op.create_index(op.f('ix_qa_log_message_id'), 'qa_log', ['message_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_qa_log_message_id'), table_name='qa_log')
    op.drop_index(op.f('ix_qa_log_created_at'), table_name='qa_log')
    op.drop_index(op.f('ix_qa_log_id'), table_name='qa_log')
    op.drop_table('qa_log')
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_column('rating')
