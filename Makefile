build:
	docker-compose build

run:
	docker-compose up

test:
	docker run --rm $(shell docker build -q .) pytest

lint:
	black app/ tests/
	isort app/ tests/
