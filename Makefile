up:
	docker compose up -d qdrant

up-full:
	docker compose --profile fullstack up --build

down:
	docker compose down

logs:
	docker compose logs -f qdrant

logs-full:
	docker compose --profile fullstack logs -f qdrant api demo
