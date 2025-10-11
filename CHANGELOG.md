# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Resolved Jinja2 TemplateNotFound error for index.html by specifying template_folder in Flask app initialization. (What: Edit to main.py; Why: Align with Dockerfile copy structure; Impact: Allows proper rendering of web pages; Links: Internal repo structure; Rollback: Revert commit and restart app).
- Fixed Ultralytics config directory warning by setting YOLO_CONFIG_DIR to a writable app path. (What: Added mkdir and export in entrypoint.sh; Why: Avoid permissions issues in container; Impact: Eliminates warning during model download; Links: https://docs.ultralytics.com/usage/cfg/ (accessed 2025-10-11T02:04:38Z); Rollback: Remove lines from entrypoint.sh).
