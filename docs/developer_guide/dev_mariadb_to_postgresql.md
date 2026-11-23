# Migration from MariaDB to PostgreSQL (2025/10/31 🎃)

## Compose configuration

Migration of production data from the original MariaDB database to a PostgreSQL database was performed locally in Docker Compose from a database backup snapshot.

The `database` service was modified to use a custom PostgreSQL container image `localhost/pgloader:dev` based on `postgres:16` and installing `pgloader`:

```dockerfile
FROM postgres:16

RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install -y --no-install-recommends \
    pgloader \
    && rm -rf /var/lib/apt/list/*
```

Also a temporary MariaDB container was added to host the import of the backup snapshot:

```yaml
  mariadb:
    image: mariadb:11.1
    platform: linux/x86_64
    restart: always
    volumes:
      - /var/tmp/shareddb:/shared
    networks:
      internal:
        aliases:
          - ${DB_HOST}
    environment:
      MARIADB_PASSWORD: ${DB_PASS}
      MARIADB_USER: ${DB_USER}
      MARIADB_DATABASE: ${DB_NAME}
      MARIADB_ALLOW_EMPTY_ROOT_PASSWORD: "true"
    profiles:
      - "full_prod"
      - "slim_prod"
      - "full_dev"
      - "slim_dev"
      - "ci"
      - "batch"

  database:
    image: localhost/pgloader:dev
    platform: linux/x86_64
    restart: always
    volumes:
      - blast-db:/var/lib/postgresql
      - /var/tmp/shareddb:/shared
    networks:
      internal:
        aliases:
          - ${DB_HOST}
    environment:
      POSTGRES_PASSWORD: ${DB_PASS}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_DB: ${DB_NAME}
    profiles:
      - "full_prod"
      - "slim_prod"
      - "full_dev"
      - "slim_dev"
      - "ci"
      - "batch"
```

## Import MariaDB backup to MariaDB

```
$ docker exec -it blast-dev-mariadb-1 bash
root@74922b45e041:/# mariadb -u"$MARIADB_USER" -p"$MARIADB_PASSWORD" $MARIADB_DATABASE < /shared/blast.649a4e7f.sql
```

## Migrate to PostgreSQL

```
$ docker exec -it blast-dev-database-1 bash

root@4e92ce8ac163:/# cat > my.load << EOF
load database from mysql://blast:password@mariadb/blast_db into pgsql://blast:password@localhost/blast_db alter schema 'blast_db' rename to 'public';
EOF

root@4e92ce8ac163:/# pgloader my.load
2025-10-31T15:49:58.010999Z LOG pgloader version "3.6.10~devel"
2025-10-31T15:49:58.030999Z LOG Migrating from #<MYSQL-CONNECTION mysql://blast@mariadb:3306/blast_db {10052AE473}>
2025-10-31T15:49:58.030999Z LOG Migrating into #<PGSQL-CONNECTION pgsql://blast@localhost:5432/blast_db {10053CDB03}>
2025-10-31T15:50:04.204845Z WARNING PostgreSQL warning: identifier "idx_17232_host_aperturephotome_transient_id_488473ab_fk_host_tran" will be truncated to "idx_17232_host_aperturephotome_transient_id_488473ab_fk_host_tr"
...
2025-10-31T15:50:20.363441Z LOG report summary reset
                                  table name     errors       rows      bytes      total time
--------------------------------------------  ---------  ---------  ---------  --------------
                             fetch meta data          0        203                     0.049s
...
                              Set Table OIDs          0         45                     0.005s
--------------------------------------------  ---------  ---------  ---------  --------------
          blast_db.host_taskregistersnapshot          0    3032341   141.2 MB         10.755s
...
--------------------------------------------  ---------  ---------  ---------  --------------
                           Total import time          ✓    7785427   591.7 MB         33.748s

```

## Dump PostgreSQL

```
root@4e92ce8ac163:/# PGPASSWORD="$DB_PASS" pg_dump --clean --no-acl --host=$DB_HOST --dbname=$DB_NAME --username=$DB_USER > /shared/pgdump.649a4e7f.sql

$ du -sh '/var/tmp/shareddb/pgdump.649a4e7f.sql' 
603M	/var/tmp/shareddb/pgdump.649a4e7f.sql

$ head '/var/tmp/shareddb/pgdump.649a4e7f.sql' 
--
-- PostgreSQL database dump
--

-- Dumped from database version 16.10 (Debian 16.10-1.pgdg13+1)
-- Dumped by pg_dump version 16.10 (Debian 16.10-1.pgdg13+1)

SET statement_timeout = 0;

$ tail '/var/tmp/shareddb/pgdump.649a4e7f.sql' 
ALTER TABLE ONLY public.silk_sqlquery
    ADD CONSTRAINT silk_sqlquery_request_id_6f8f0527_fk_silk_request_id FOREIGN KEY (request_id) REFERENCES public.silk_request(id) DEFERRABLE INITIALLY DEFERRED;


--
-- PostgreSQL database dump complete
--

```

## Test restore from PostgreSQL dump

Stop all the containers `docker compose down` and purge the persistent volumes, then relaunch the application to reinitialize a fresh database.
Open a shell in the postgres container and run the restore command twice:

```
PGPASSWORD="${DB_PASS}" psql \
--host=$DB_HOST \
--dbname=$DB_NAME \
--username=$DB_USER < /shared/pgdump.649a4e7f.sql
```
