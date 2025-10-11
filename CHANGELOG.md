# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Introduced RTSP reachability pre-checks and offline placeholder frames when streams are unavailable. (What: Added connectivity helper and offline frame generation in application code and templates; Why: Avoid UI freezes when the configured RTSP stream is offline; Impact: Users receive immediate feedback while keeping the app responsive; Links: Internal issue report (accessed 2025-10-11T03:21:46Z); Rollback: Revert connectivity helper and placeholder changes).

### Changed
- Normalized polygon configuration handling and moved polygon formatting logic into the Flask view to prevent template syntax errors. (What: Adjusted config serialization/deserialization and view context data; Why: Ensure `/config` renders even with complex polygon data; Impact: Stable configuration page rendering; Links: Internal issue report (accessed 2025-10-11T03:21:46Z); Rollback: Revert config handling and template updates).

### Removed
- Removed the health check navigation link from the landing page while keeping the endpoint accessible. (What: Updated index template; Why: Link provided no additional value and was confusing; Impact: Cleaner UI without losing functionality; Links: Internal issue report (accessed 2025-10-11T03:21:46Z); Rollback: Revert template change).

### Fixed
- Resolved Jinja2 TemplateNotFound error for index.html by specifying template_folder in Flask app initialization. (What: Edit to main.py; Why: Align with Dockerfile copy structure; Impact: Allows proper rendering of web pages; Links: Internal repo structure; Rollback: Revert commit and restart app).
- Fixed Ultralytics config directory warning by setting YOLO_CONFIG_DIR to a writable app path. (What: Added mkdir and export in entrypoint.sh; Why: Avoid permissions issues in container; Impact: Eliminates warning during model download; Links: https://docs.ultralytics.com/usage/cfg/ (accessed 2025-10-11T02:04:38Z); Rollback: Remove lines from entrypoint.sh).
