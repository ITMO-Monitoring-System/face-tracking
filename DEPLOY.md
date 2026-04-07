# Face Tracking deploy

Push в `main` запускает workflow `.github/workflows/deploy.yml`.

## Required GitHub secrets

- `DEPLOY_HOST` (или `SSH_HOST`)
- `DEPLOY_PORT` (или `SSH_PORT`, по умолчанию `22`)
- `DEPLOY_USERNAME` (или `SSH_USERNAME`)
- `DEPLOY_SSH_PRIVATE_KEY` (или `SSH_PRIVATE_KEY`)
- `DEPLOY_PATH` (опционально, по умолчанию `/opt/fizon/face-tracking`)
- `APP_ENV_FILE` (опционально, содержимое `.env.production`)

Если `APP_ENV_FILE` не задан, используются значения по умолчанию из `docker-compose.prod.yml`.
