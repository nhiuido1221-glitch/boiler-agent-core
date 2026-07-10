-- ==============================================================================
-- Supabase schema cho Boiler Agent Core
-- Chay 1 lan trong Supabase SQL Editor (Project -> SQL Editor -> New query).
-- ==============================================================================

-- ---------- Bang log tuong tac (Phase 4) ----------
create table if not exists boiler_agent_logs (
    id              bigint generated always as identity primary key,
    group_id        text,
    project_id      text,
    user_role       text,
    raw_message     text,
    final_response  text,
    is_emergency    boolean default false,
    keywords_found  text[] default '{}',
    routing_log     text[] default '{}',
    rag_sources     jsonb default '[]'::jsonb,
    created_at      timestamptz not null default now()
);

create index if not exists idx_boiler_agent_logs_group_id on boiler_agent_logs (group_id);
create index if not exists idx_boiler_agent_logs_created_at on boiler_agent_logs (created_at desc);
create index if not exists idx_boiler_agent_logs_is_emergency on boiler_agent_logs (is_emergency) where is_emergency = true;

alter table boiler_agent_logs enable row level security;

drop policy if exists "service_role_full_access" on boiler_agent_logs;

create policy "service_role_full_access"
    on boiler_agent_logs
    for all
    to service_role
    using (true)
    with check (true);

-- ---------- Bang anh xa Telegram group -> du an (Cai tien: Multi-tenant RAG) ----------
create table if not exists boiler_projects (
    project_id      text not null,
    project_name    text not null,
    group_id        text primary key,
    created_by      text,
    boiler_type     text,
    created_at      timestamptz not null default now()
);

-- Neu bang da ton tai tu truoc (chua co cot boiler_type), them cot rieng:
alter table boiler_projects add column if not exists boiler_type text;

create index if not exists idx_boiler_projects_project_id on boiler_projects (project_id);

alter table boiler_projects enable row level security;

drop policy if exists "service_role_full_access" on boiler_projects;

create policy "service_role_full_access"
    on boiler_projects
    for all
    to service_role
    using (true)
    with check (true);
