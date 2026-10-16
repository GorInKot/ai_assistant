"""add portal_section to qa_log and page_label to messages

Вопросы из панели на страницах портала: в журнале вопросов — раздел портала (сколько и о чём
спрашивают в каждом разделе, помогают ли ответы), в истории беседы — короткая метка раздела
у вопроса. Текст страницы не хранится. Колонки необязательные, данные не затрагиваются.

Revision ID: c9e2a7f43d18
Revises: b7d2e4f91a63
Create Date: 2026-10-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9e2a7f43d18'
down_revision: Union[str, Sequence[str], None] = 'b7d2e4f91a63'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('qa_log', sa.Column('portal_section', sa.String(), nullable=True))
    op.create_index('ix_qa_log_portal_section', 'qa_log', ['portal_section'])
    op.add_column('messages', sa.Column('page_label', sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_column('page_label')
    op.drop_index('ix_qa_log_portal_section', table_name='qa_log')
    with op.batch_alter_table('qa_log', schema=None) as batch_op:
        batch_op.drop_column('portal_section')
