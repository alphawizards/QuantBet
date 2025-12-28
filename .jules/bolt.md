## 2024-05-22 - Backend JSON Serialization Optimization
**Learning:** Default `JSONResponse` in FastAPI can be a bottleneck for large JSON payloads. Using `ORJSONResponse` with `orjson` library provides significantly faster serialization (benchmark showed ~2x speedup for 10k items).
**Action:** For high-throughput APIs returning large JSON structures, always prefer `ORJSONResponse` over the default. Set `default_response_class=ORJSONResponse` in FastAPI app initialization.
