# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


### Added
- Added real-time proximity metric panel replacing manual max-distance control, with automatic polygon-based distance scaling and live updates. (What: Frontend UI card, backend metric computation, and SSE/polling endpoint; Why: Provide clear, continuously updated proximity information derived from polygon geometry; Impact: Users monitor proximity without manual tuning; Links: User request (accessed 2025-10-11T05:08:53Z); Rollback: Revert proximity UI/metric changes).
- Added optional TensorRT inference path activated via ENABLE_TRT, including automated ONNX→TensorRT engine creation and person-class filtering. (What: entrypoint.sh conversion flow, model loader adjustments; Why: Improve NVIDIA GPU utilization when TensorRT is available; Impact: Faster, GPU-optimized inference with fallback to PyTorch; Links: User request (accessed 2025-10-11T05:08:53Z); Rollback: Disable ENABLE_TRT and revert loader changes).
- Added multithreaded detection worker to keep health and metrics endpoints responsive while streaming. (What: Detection thread, shared state synchronization; Why: Prevent UI/API blocking under heavy inference load; Impact: Stable health checks and metrics even when video stream is active; Links: User request (accessed 2025-10-11T05:08:53Z); Rollback: Restore single-threaded loop).
### Changed
- Replaced manual max-distance configuration with automatic polygon-derived scaling and enforced target point validation. (What: Config schema updates, validation logic, UI adjustments; Why: Reduce misconfiguration errors and align proximity score with actual detection geometry; Impact: Configuration simplified and more accurate metrics; Links: User request (accessed 2025-10-11T05:08:53Z); Rollback: Reinstate max-distance field and prior validation).
- Locked inference frame rate to 5 FPS by dropping excess frames before processing. (What: Frame throttling logic in detection loop; Why: Balance performance and resource utilization; Impact: Predictable load and smoother GPU usage; Links: User request (accessed 2025-10-11T05:08:53Z); Rollback: Remove frame throttling).
- Updated docker-compose to set ENABLE_TRT=1 by default and refreshed README usage instructions. (What: docker-compose.yml, README edits; Why: Promote GPU-accelerated mode and correct startup guidance; Impact: Users start stack with proper defaults; Links: User request (accessed 2025-10-11T05:08:53Z); Rollback: Restore previous compose/env docs).
### Fixed
- Injected datetime helper into template context to resolve Jinja undefined error in footer. (What: main.py context update; Why: Prevent 500 error when rendering index; Impact: Landing page loads successfully; Links: Runtime log (accessed 2025-10-11T05:08:53Z); Rollback: Remove context injection).
### Removed
- Removed obsolete max-distance field from configuration schema and example file. (What: config.json.example updates, form adjustments; Why: Feature superseded by automatic scaling; Impact: Cleaner config defaults; Links: User request (accessed 2025-10-11T05:08:53Z); Rollback: Reintroduce max-distance field).


### Added
- Introduced RTSP reachability pre-checks and offline placeholder frames when streams are unavailable. (What: Added connectivity helper and offline frame generation in application code and templates; Why: Avoid UI freezes when the configured RTSP stream is offline; Impact: Users receive immediate feedback while keeping the app responsive; Links: Internal issue report (accessed 2025-10-11T03:21:46Z); Rollback: Revert connectivity helper and placeholder changes).
- Enabled inline editing of stream URL, detection polygon, and target point directly on the main page via an interactive overlay. (What: Replace separate config page with single-page configuration UI; Why: Simplify UX when the stream is offline and `/config` route is unreachable; Impact: Users can tweak settings without leaving the main view; Links: Internal follow-up request (accessed 2025-10-11T04:06:34Z); Rollback: Revert combined UI changes).

### Changed
- Replaced `/config` navigation flow with inline controls and edit mode toggle on the landing page. (What: Home template/form adjustments and AJAX handlers; Why: Clicking `Config` previously failed when the stream was unresponsive; Impact: Users manage configuration in-place; Links: Internal follow-up request (accessed 2025-10-11T04:06:34Z); Rollback: Restore prior templates and routes).
- Updated README with clarified app purpose and usage instructions while removing redundant sandbox warning. (What: Documentation edits; Why: Align docs with new UX and features; Impact: Clearer onboarding instructions; Links: Internal follow-up request (accessed 2025-10-11T04:06:34Z); Rollback: Revert README changes).
- Normalized polygon configuration handling and moved polygon formatting logic into the Flask view to prevent template syntax errors. (What: Adjusted config serialization/deserialization and view context data; Why: Ensure `/config` renders even with complex polygon data; Impact: Stable configuration page rendering; Links: Internal issue report (accessed 2025-10-11T03:21:46Z); Rollback: Revert config handling and template updates).

### Removed
- Removed the health check navigation link from the landing page while keeping the endpoint accessible. (What: Updated index template; Why: Link provided no additional value and was confusing; Impact: Cleaner UI without losing functionality; Links: Internal issue report (accessed 2025-10-11T03:21:46Z); Rollback: Revert template change).

### Fixed
- Resolved Jinja2 TemplateNotFound error for index.html by specifying template_folder in Flask app initialization. (What: Edit to main.py; Why: Align with Dockerfile copy structure; Impact: Allows proper rendering of web pages; Links: Internal repo structure; Rollback: Revert commit and restart app).
- Fixed Ultralytics config directory warning by setting YOLO_CONFIG_DIR to a writable app path. (What: Added mkdir and export in entrypoint.sh; Why: Avoid permissions issues in container; Impact: Eliminates warning during model download; Links: https://docs.ultralytics.com/usage/cfg/ (accessed 2025-10-11T02:04:38Z); Rollback: Remove lines from entrypoint.sh).
