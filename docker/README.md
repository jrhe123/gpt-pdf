#### Dockerhub:

- https://hub.docker.com/_/postgres

  - docker pull postgres

- https://hub.docker.com/r/ankane/pgvector
  - docker pull ankane/pgvector

#### Run it

- docker run --name pgsql -e POSTGRES_PASSWORD=test -p 5432:5432 -d ankane/pgvector
