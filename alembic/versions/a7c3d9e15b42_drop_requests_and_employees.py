"""drop requests and employees

Заявки и маршрутизация ответственным переехали в Портал; ассистент больше не
оформляет заявки. Удаляем связанные таблицы: заявки с историей, каталог типов и
слоты, сотрудников, области ответственности и привязки.

ВНИМАНИЕ: upgrade необратимо удаляет данные этих таблиц. downgrade воссоздаёт
только схему (пустые таблицы).

Revision ID: a7c3d9e15b42
Revises: 21414f229b8f
Create Date: 2026-10-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7c3d9e15b42'
down_revision: Union[str, Sequence[str], None] = '21414f229b8f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop request/employee tables (дочерние — раньше родительских)."""
    op.drop_table('request_events')
    op.drop_table('responsibilities')
    op.drop_table('requests')
    op.drop_table('request_type_slots')
    op.drop_table('employees')
    op.drop_table('responsibility_areas')
    op.drop_table('request_types')


def downgrade() -> None:
    """Воссоздать схему удалённых таблиц (данные не восстанавливаются)."""
    op.create_table('request_types',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('type_slug', sa.String(), nullable=False),
    sa.Column('title', sa.String(), nullable=False),
    sa.Column('responsibility_area_slug', sa.String(), nullable=False),
    sa.Column('is_anonymous', sa.Boolean(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('trigger_keywords_json', sa.Text(), nullable=False),
    sa.Column('examples_json', sa.Text(), nullable=False),
    sa.Column('sort_order', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('request_types', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_request_types_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_request_types_responsibility_area_slug'), ['responsibility_area_slug'], unique=False)
        batch_op.create_index(batch_op.f('ix_request_types_type_slug'), ['type_slug'], unique=True)
    op.create_table('responsibility_areas',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('slug', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('responsibility_areas', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_responsibility_areas_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_responsibility_areas_slug'), ['slug'], unique=True)
    op.create_table('employees',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('email', sa.String(), nullable=False),
    sa.Column('full_name', sa.String(), nullable=False),
    sa.Column('position', sa.String(), nullable=True),
    sa.Column('division', sa.String(), nullable=True),
    sa.Column('subdivision', sa.String(), nullable=True),
    sa.Column('phone', sa.String(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('employees', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_employees_division'), ['division'], unique=False)
        batch_op.create_index(batch_op.f('ix_employees_email'), ['email'], unique=True)
        batch_op.create_index(batch_op.f('ix_employees_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_employees_subdivision'), ['subdivision'], unique=False)
        batch_op.create_index(batch_op.f('ix_employees_user_id'), ['user_id'], unique=False)
    op.create_table('request_type_slots',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('request_type_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('question', sa.String(), nullable=False),
    sa.Column('required', sa.Boolean(), nullable=False),
    sa.Column('sort_order', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['request_type_id'], ['request_types.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('request_type_slots', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_request_type_slots_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_request_type_slots_request_type_id'), ['request_type_id'], unique=False)
    op.create_table('requests',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('type_slug', sa.String(), nullable=False),
    sa.Column('type_title', sa.String(), nullable=False),
    sa.Column('requester_user_id', sa.Integer(), nullable=True),
    sa.Column('assigned_employee_id', sa.Integer(), nullable=True),
    sa.Column('conversation_id', sa.Integer(), nullable=True),
    sa.Column('is_anonymous', sa.Boolean(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('payload_json', sa.Text(), nullable=False),
    sa.Column('summary', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['assigned_employee_id'], ['employees.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['requester_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('requests', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_requests_assigned_employee_id'), ['assigned_employee_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_requests_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_requests_requester_user_id'), ['requester_user_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_requests_status'), ['status'], unique=False)
        batch_op.create_index(batch_op.f('ix_requests_type_slug'), ['type_slug'], unique=False)
    op.create_table('responsibilities',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('employee_id', sa.Integer(), nullable=False),
    sa.Column('area_id', sa.Integer(), nullable=False),
    sa.Column('scope_division', sa.String(), nullable=True),
    sa.Column('scope_subdivision', sa.String(), nullable=True),
    sa.Column('is_primary', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['area_id'], ['responsibility_areas.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['employee_id'], ['employees.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('employee_id', 'area_id', 'scope_division', 'scope_subdivision', name='uq_responsibility_scope')
    )
    with op.batch_alter_table('responsibilities', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_responsibilities_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_responsibilities_scope_division'), ['scope_division'], unique=False)
        batch_op.create_index(batch_op.f('ix_responsibilities_scope_subdivision'), ['scope_subdivision'], unique=False)
    op.create_table('request_events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('request_id', sa.Integer(), nullable=False),
    sa.Column('event_type', sa.String(), nullable=False),
    sa.Column('actor_user_id', sa.Integer(), nullable=True),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['actor_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['request_id'], ['requests.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('request_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_request_events_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_request_events_request_id'), ['request_id'], unique=False)

