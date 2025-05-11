# Target to install dependencies from requirements.txt
install:
	pip install -r requirements.txt


# Target to clean up temporary files and __pycache__ directories
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache
	


# Target to format code using black (assuming you have black installed)
format:
	black .

# Phony targets (targets that are not files)
.PHONY: install clean format lint typecheck