#!/bin/sh

echo "⏳ Waiting for MySQL at $DATABASE_HOST:$DATABASE_PORT..."

while ! nc -z "$DATABASE_HOST" "$DATABASE_PORT"; do
  sleep 1
done

echo "✅ MySQL is up and running!"

exec "$@"