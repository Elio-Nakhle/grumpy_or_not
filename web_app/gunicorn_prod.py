import os

grumpy_port = int(os.environ.get("GRUMPY_PORT", 8080))
grumpy_host = os.environ.get("GRUMPY_HOST", "0.0.0.0")

log_level = "info"
bind = f"{grumpy_host}:{grumpy_port}"
workers = 4
