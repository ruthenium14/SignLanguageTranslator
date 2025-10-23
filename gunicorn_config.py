# gunicorn_config.py
import multiprocessing

# Memory-optimized configuration for Render free tier
workers = 1  # Only 1 worker to minimize memory usage
worker_class = 'sync'
worker_connections = 50
timeout = 120  # Increased timeout for ML processing
keepalive = 5
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 20
graceful_timeout = 30
preload_app = False  # Don't preload to save startup memory

# Bind to the PORT environment variable (Render requirement)
import os
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
