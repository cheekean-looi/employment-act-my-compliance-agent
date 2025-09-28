#!/usr/bin/env python3
"""
Production gunicorn configuration for Employment Act Malaysia compliance agent.
Optimized for production deployment with 3 workers and async support.
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('API_PORT', '8001')}"
backlog = 2048

# Worker processes
workers = 3  # Recommended for production workload
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings
timeout = 60  # Match vLLM client timeout
keepalive = 5
graceful_timeout = 30

# Process naming
proc_name = "employment-act-api"

# Logging
loglevel = "info"
accesslog = "-"  # stdout
errorlog = "-"   # stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Security
limit_request_line = 8192
limit_request_fields = 100
limit_request_field_size = 8190

# Preload application for better memory usage
preload_app = True

# Environment variables
raw_env = [
    f"PYTHONPATH={os.getcwd()}",
    f"API_PORT={os.getenv('API_PORT', '8001')}",
    f"VLLM_URL={os.getenv('VLLM_URL', 'http://localhost:8000')}",
    f"REDIS_URL={os.getenv('REDIS_URL', '')}",
]

# Enable async support and optimize for FastAPI
worker_tmp_dir = "/dev/shm"  # Use shared memory for better performance

def when_ready(server):
    """Called when server is ready to accept connections."""
    server.log.info("Employment Act API server ready")

def on_exit(server):
    """Called when server is shutting down."""
    server.log.info("Employment Act API server shutting down")

def worker_int(worker):
    """Called when worker receives SIGINT or SIGQUIT signal."""
    worker.log.info(f"Worker {worker.pid} received interrupt signal")

def pre_fork(server, worker):
    """Called before forking a worker."""
    server.log.info(f"Forking worker {worker.age}")

def post_fork(server, worker):
    """Called after worker has been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def post_worker_init(worker):
    """Called after worker has initialized the application."""
    worker.log.info(f"Worker {worker.pid} initialized")

def worker_abort(worker):
    """Called when worker is aborted."""
    worker.log.error(f"Worker {worker.pid} aborted")